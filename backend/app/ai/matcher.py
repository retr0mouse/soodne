import logging
import re
import requests
import numpy as np
import torch
import xgboost as xgb
import faiss
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from torchvision import models, transforms
from sqlalchemy import func
from typing import Optional, List, Tuple
from sqlalchemy import or_
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel

from app.models.product_store_data import ProductStoreData
from app.models.product_matching_log import ProductMatchingLog
from app.models.product import Product
from app.models.unit import Unit
from app.core.logger import setup_logger

# Replace the existing logger setup
logger = setup_logger("app.ai.matcher")

# ------------------ TEXT & IMAGE MODELS ------------------

# Replace the existing sentence model with EstBERT for Estonian language
class EstBERTSentenceTransformer:
    def __init__(self, model_name="tartuNLP/EstBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Set model to evaluation mode
        self.model.eval()
        self.output_dim = 768  # EstBERT's output dimension
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts, batch_size=32):
        """
        Encode texts to vector embeddings
        """
        try:
            # Convert string to list if it's a single text
            if isinstance(texts, str):
                texts = [texts]
                
            all_embeddings = []
            
            # Process in batches for memory efficiency
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
                
                # Get embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                
                # Pool to get sentence embeddings
                sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                
                all_embeddings.append(sentence_embeddings.numpy())
                
            # Concatenate batches
            if len(all_embeddings) > 0:
                all_embeddings = np.vstack(all_embeddings)
                
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in EstBERT encoding: {str(e)}")
            # Return zero vector with correct dimensionality
            return np.zeros((len(texts), self.output_dim), dtype=np.float32)

# Initialize both models - general multilingual and Estonian-specific
general_sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
estonian_sentence_model = EstBERTSentenceTransformer()

# Cache the model dimensions
ESTBERT_DIM = 768  # EstBERT's output dimension
GENERAL_DIM = 384  # all-MiniLM-L6-v2's output dimension

image_model = models.resnet50(weights="IMAGENET1K_V2")
image_model.eval()

image_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

xgb_clf: Optional[xgb.XGBClassifier] = None

# ------------------ LOAD ML CLASSIFIER ------------------

def load_xgb_model(path: str) -> None:
    global xgb_clf
    m = xgb.XGBClassifier()
    m.load_model(path)
    xgb_clf = m

# ------------------ TEXT & IMAGE PROCESSING ------------------

def preprocess_text(t: str) -> str:
    if not t:
        return ""
        
    # Convert to lowercase
    t = t.lower()
    
    # Remove store-specific prefixes like "PT" (Prisma/Rimi specific)
    t = re.sub(r'^pt\s+', '', t)
    
    # Remove numbers and units at the end (e.g., "500g", "1kg")
    t = re.sub(r'\d+\s*(?:g|kg|l|ml|tk)$', '', t)
    
    # Remove brand indicators like "brand:" or "by:"
    t = re.sub(r'(brand|by)\s*:\s*[\w\s]+', '', t)
    
    # Remove extra spaces
    t = re.sub(r'\s+', ' ', t).strip()
    
    return t

def get_text_embedding(t: str) -> np.ndarray:
    """
    Get text embedding using the most appropriate model for the input text.
    Returns a normalized embedding vector of consistent dimensionality (768).
    """
    try:
        if not t:
            return np.zeros(ESTBERT_DIM, dtype=np.float32)
        
        # For short texts, always use EstBERT to ensure consistency
        if len(t) < 5:
            emb = estonian_sentence_model.encode(t)[0]
            return emb
        
        # Get embeddings from both models
        try:
            est_embedding = estonian_sentence_model.encode(t)[0]  # 768-dim
        except Exception as e:
            logger.error(f"EstBERT embedding failed: {str(e)}")
            est_embedding = np.zeros(ESTBERT_DIM, dtype=np.float32)
            
        try:
            gen_embedding = general_sentence_model.encode(t)  # 384-dim
            # Pad general embedding to match EstBERT dimensionality
            gen_embedding_padded = np.pad(gen_embedding, (0, ESTBERT_DIM - GENERAL_DIM), 'constant')
        except Exception as e:
            logger.error(f"General model embedding failed: {str(e)}")
            gen_embedding_padded = np.zeros(ESTBERT_DIM, dtype=np.float32)
        
        # Compute confidence scores for each model
        est_confidence = np.std(est_embedding) * np.max(np.abs(est_embedding))
        gen_confidence = np.std(gen_embedding) * np.max(np.abs(gen_embedding))
        
        # Select the model with higher confidence
        if est_confidence > gen_confidence:
            logger.debug(f"Selected EstBERT model for '{t}'")
            final_embedding = est_embedding
        else:
            logger.debug(f"Selected general model for '{t}'")
            final_embedding = gen_embedding_padded
            
        # Ensure the output is normalized and has the correct shape
        final_embedding = final_embedding.reshape(ESTBERT_DIM)
        norm = np.linalg.norm(final_embedding)
        if norm > 1e-8:
            final_embedding = final_embedding / norm
            
        return final_embedding
        
    except Exception as e:
        logger.error(f"Error in get_text_embedding for text '{t}': {str(e)}")
        return np.zeros(ESTBERT_DIM, dtype=np.float32)

def get_image_embedding_from_url(url: str) -> np.ndarray:
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        i = Image.open(BytesIO(r.content)).convert("RGB")
        t = image_preprocess(i).unsqueeze(0)
        with torch.no_grad():
            f = image_model(t).numpy().flatten()
        return f
    except:
        return np.zeros(1000, dtype=np.float32)

# ------------------ ATTRIBUTE COMPARISON ------------------

def compare_weight(a: Optional[str], b: Optional[str], 
                  unit_a: Optional['Unit'] = None, 
                  unit_b: Optional['Unit'] = None) -> float:
    if not a or not b:
        return 0.0
    
    # Convert inputs to strings if they aren't already
    a_str = str(a) if a is not None else ""
    b_str = str(b) if b is not None else ""
    
    def parse_with_unit(x: str) -> Tuple[Optional[float], str]:
        # Extract number and unit
        match = re.search(r'([\d.,]+)\s*(ml|l|g|kg)?', x.lower())
        if not match:
            return None, ""
            
        value_str, unit = match.groups()
        try:
            value = float(value_str.replace(',', '.'))
            return value, unit or ""
        except:
            return None, ""
            
    val_a, unit_a_str = parse_with_unit(a_str)
    val_b, unit_b_str = parse_with_unit(b_str)
    
    if val_a is None or val_b is None:
        return 0.0
        
    # Convert to base units (ml for liquids, g for weight)
    def normalize_value(val: float, unit: str) -> float:
        if unit in ['l']:
            return val * 1000  # Convert to ml
        if unit in ['kg']:
            return val * 1000  # Convert to g
        return val
        
    val_a = normalize_value(val_a, unit_a_str)
    val_b = normalize_value(val_b, unit_b_str)
    
    # Compare with percentage tolerance
    try:
        if val_a == 0 and val_b == 0:
            return 100.0
        if val_a == 0 or val_b == 0:
            return 0.0
        
        # Calculate percentage difference
        diff = abs(val_a - val_b) / max(val_a, val_b)
        
        # If difference is within 5%, return full score
        if diff <= 0.05:
            return 100.0
            
        # If difference is within 20%, return partial score
        if diff <= 0.20:
            return (1 - diff) * 100
            
    except ZeroDivisionError:
        logger.error("Zero division error in weight comparison")
        return 0.0
        
    return 0.0

# ------------------ FEATURE EXTRACTION ------------------

def compute_features(n1: str, n2: str, w1: Optional[str], w2: Optional[str], 
                     u1: Optional['Unit'] = None, u2: Optional['Unit'] = None,
                     i1: Optional[np.ndarray] = None, i2: Optional[np.ndarray] = None) -> dict:
    # Semantic text similarity using embeddings
    try:
        # Preprocess text for better semantic understanding
        p1 = preprocess_text(n1)
        p2 = preprocess_text(n2)
        
        # Get semantic embeddings for full text
        full_e1 = get_text_embedding(p1)
        full_e2 = get_text_embedding(p2)
        
        # Calculate overall semantic similarity
        full_semantic_similarity = float(util.cos_sim(full_e1, full_e2)[0][0] * 100.0)
        
        # Word-by-word semantic analysis
        # Split product names into individual words
        words1 = [w for w in p1.split() if len(w) > 2]  # Filter out short words
        words2 = [w for w in p2.split() if len(w) > 2]
        
        # Calculate key word matching score
        key_matches = 0
        total_words = max(len(words1), len(words2))
        
        if total_words > 0:
            # Find best matching words between the two product names
            for word1 in words1:
                word1_emb = get_text_embedding(word1)
                best_match_score = 0
                
                for word2 in words2:
                    word2_emb = get_text_embedding(word2)
                    match_score = float(util.cos_sim(word1_emb, word2_emb)[0][0])
                    
                    # If the words are almost identical, they're a match
                    if match_score > 0.9:
                        best_match_score = match_score
                        break
                    
                    # Otherwise keep track of best partial match
                    best_match_score = max(best_match_score, match_score)
                
                # If best match is high, count it as a key word match
                if best_match_score > 0.85:
                    key_matches += best_match_score
            
            # Calculate percentage of matching key words
            key_word_similarity = (key_matches / total_words) * 100.0
        else:
            key_word_similarity = 0.0
        
        # Calculate different product indicator score
        # This helps identify if products are different types despite similar patterns
        different_indicator = 0.0
        
        # If any significant word appears in one product but not in the other,
        # and it's a product-defining word, increase the different indicator
        product_defining_words = []
        for word in words1 + words2:
            # Check if word has high semantic similarity with any product-defining term
            word_emb = get_text_embedding(word)
            food_emb = get_text_embedding("food")
            
            # If this word has high similarity with product categories
            # but appears in only one of the products, they're likely different
            if word in words1 and word not in words2 or word in words2 and word not in words1:
                different_indicator += 25.0
        
        # Combined semantic similarity that weights both approaches
        # Word-level analysis gets higher weight to better distinguish similar products
        semantic_similarity = full_semantic_similarity * 0.3 + key_word_similarity * 0.7
        
        # Adjust for different product indicators
        semantic_similarity = max(0, semantic_similarity - different_indicator)
        
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        semantic_similarity = 0.0
    
    # Weight similarity 
    weight_similarity = compare_weight(w1, w2, u1, u2)
    
    # Image similarity if available
    image_similarity = 0.0
    if i1 is not None and i2 is not None:
        try:
            n1_ = np.linalg.norm(i1)
            n2_ = np.linalg.norm(i2)
            if n1_ > 1e-8 and n2_ > 1e-8:
                image_similarity = float(np.dot(i1, i2) / (n1_ * n2_) * 100.0)
        except Exception as e:
            logger.error(f"Error computing image similarity: {e}")
    
    return {
        "semantic_similarity": semantic_similarity,
        "weight_similarity": weight_similarity,
        "image_similarity": image_similarity
    }

def heuristic_score(f: dict) -> float:
    return f["text_cos_sim"] * 0.5 + f["weight_sim"] * 0.25 + f["image_sim"] * 0.25

def ml_match_score(f: dict) -> float:
    if xgb_clf is None:
        # Dynamic weights for scoring based on feature quality
        # Calculate feature confidence based on their values
        features = [f["semantic_similarity"], f["weight_similarity"], f["image_similarity"]]
        feature_sum = sum(max(0, x) for x in features)
        
        if feature_sum < 1e-6:
            return 0.0
        
        # Normalize and compute dynamic weights
        weights = []
        for value in features:
            # Higher values get higher confidence/weight
            value = max(0, value)
            weight = (value / feature_sum) if feature_sum > 0 else 0
            # Apply a non-linear transformation to increase weight difference
            weight = weight ** 0.7  # This increases the impact of stronger features
            weights.append(weight)
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        else:
            # Fallback if all weights are zero
            weights = [0.8, 0.15, 0.05]
        
        # Calculate score based on dynamic weights
        score = sum(value * weight for value, weight in zip(features, weights))
        
        # Boost score for very strong matches in any feature
        max_feature = max(features)
        if max_feature > 90:
            boost = (max_feature - 90) * 0.5
            score = min(100, score + boost)
            
        return max(0, score)  # Ensure score isn't negative
    
    # ML-based scoring if classifier is available
    v = np.array([f["semantic_similarity"], f["weight_similarity"], f["image_similarity"]], 
                 dtype=np.float32).reshape(1, -1)
    p = xgb_clf.predict_proba(v)[0][1]
    return float(p * 100.0)

# Add key product word detection function
def extract_key_product_words(product_name: str) -> List[str]:
    """
    Extract the key words that define what the product actually is.
    Examples: in "Frozen strawberries Rimi 400g", "strawberries" is the key word.
              in "Chicken soup with vegetables", "chicken" and "soup" are key words.
    
    Returns a list of key product-defining words.
    """
    # Preprocess the text
    clean_text = preprocess_text(product_name)
    words = clean_text.split()
    
    if not words:
        return []
    
    # Create embeddings for all meaningful words
    all_word_embeddings = [(word, get_text_embedding(word)) for word in words if len(word) > 2]
    
    if not all_word_embeddings:
        return []
    
    # Instead of comparing to hardcoded category list, use statistical approach
    # Calculate the importance of each word based on its embedding properties
    word_importance = {}
    
    # Compute embedding statistics
    all_embeddings = [emb for _, emb in all_word_embeddings]
    centroid = np.mean(all_embeddings, axis=0)
    
    # Compute importance by measuring how much each word contributes to overall meaning
    for word, emb in all_word_embeddings:
        # Words that are more distinct/specific tend to be more important product identifiers
        # Calculate distance from embedding centroid (more distant = more distinctive)
        distance_from_centroid = np.linalg.norm(emb - centroid)
        
        # Calculate embedding magnitude (stronger magnitude = more semantically significant)
        magnitude = np.linalg.norm(emb)
        
        # Calculate standard deviation of embedding (higher variation = more information)
        std_dev = np.std(emb)
        
        # Combined importance score
        importance = distance_from_centroid * magnitude * std_dev
        word_importance[word] = importance
    
    # Sort words by importance
    sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top words as key product words
    # Dynamically select number of keywords based on name length and word distribution
    num_keywords = max(1, min(len(sorted_words) // 2, 3))
    
    return [word for word, _ in sorted_words[:num_keywords]]

def check_product_compatibility(n1: str, n2: str) -> Tuple[bool, float]:
    """
    Check if two products are compatible by analyzing their key product words.
    Returns a tuple (is_compatible, compatibility_score).
    """
    # Extract key product words
    key_words1 = extract_key_product_words(n1)
    key_words2 = extract_key_product_words(n2)
    
    logger.debug(f"Key words for '{n1}': {key_words1}")
    logger.debug(f"Key words for '{n2}': {key_words2}")
    
    if not key_words1 or not key_words2:
        # If we couldn't extract key words, we can't determine incompatibility
        return True, 50.0
    
    # Get embeddings for all key words
    key_embs1 = [get_text_embedding(word) for word in key_words1]
    key_embs2 = [get_text_embedding(word) for word in key_words2]
    
    # Find the best matching key word between products
    best_match_score = 0.0
    
    for emb1 in key_embs1:
        for emb2 in key_embs2:
            similarity = float(util.cos_sim(emb1, emb2)[0][0])
            best_match_score = max(best_match_score, similarity)
    
    # Convert to percentage score
    compatibility_score = best_match_score * 100.0
    
    # Dynamically determine compatibility threshold based on word distribution
    # Calculate average similarity within each product's key words
    within_product1_sim = 0.0
    within_product2_sim = 0.0
    
    # Calculate internal similarity for product 1
    if len(key_embs1) > 1:
        count = 0
        for i in range(len(key_embs1)):
            for j in range(i+1, len(key_embs1)):
                within_product1_sim += float(util.cos_sim(key_embs1[i], key_embs1[j])[0][0])
                count += 1
        within_product1_sim = within_product1_sim / count if count > 0 else 0.0
    
    # Calculate internal similarity for product 2
    if len(key_embs2) > 1:
        count = 0
        for i in range(len(key_embs2)):
            for j in range(i+1, len(key_embs2)):
                within_product2_sim += float(util.cos_sim(key_embs2[i], key_embs2[j])[0][0])
                count += 1
        within_product2_sim = within_product2_sim / count if count > 0 else 0.0
    
    # Average internal similarity
    avg_internal_sim = (within_product1_sim + within_product2_sim) / 2.0
    
    # Dynamic threshold: products should be at least as similar between them as they are internally
    # but with a minimum reasonable baseline
    threshold = max(0.35, avg_internal_sim) * 100.0
    
    # Products are compatible if their key words are sufficiently similar
    is_compatible = compatibility_score > threshold
    
    logger.debug(f"Product compatibility: {compatibility_score:.1f}% vs threshold {threshold:.1f}% - {'Compatible' if is_compatible else 'Incompatible'}")
    
    return is_compatible, compatibility_score

def match_products(n1: str, n2: str, w1: Optional[str], w2: Optional[str], 
                   u1: Optional[str], u2: Optional[str],
                   unit1: Optional['Unit'] = None, unit2: Optional['Unit'] = None,
                   e1: Optional[str] = None, e2: Optional[str] = None,
                   t: float = 82.0) -> Tuple[bool, float]:
    """
    Match products based primarily on semantic understanding via embeddings.
    This approach relies on the model's ability to understand product meanings
    rather than hardcoded rules.
    """
    # First check EAN codes if available
    if e1 and e2 and e1 == e2:
        return True, 100.0
    
    # Check if products are fundamentally compatible types
    # This prevents matching completely different product categories
    is_compatible, compatibility_score = check_product_compatibility(n1, n2)
    
    if not is_compatible:
        logger.info(f"Products are incompatible types: '{n1}' vs '{n2}'")
        return False, min(compatibility_score, t/2.0)  # Dynamic limit relative to threshold
    
    # If product names contain numbers that don't match, they're likely different products
    # This helps distinguish between similar products with different numeric attributes
    num_pattern = r'\d+'
    nums1 = re.findall(num_pattern, n1)
    nums2 = re.findall(num_pattern, n2)
    
    # Check if both products have numbers but they don't match
    # Exclude weight numbers that are likely to be in a known range (e.g., 400g)
    weight_pattern = r'\d+\s*(?:g|kg|ml|l)'
    weight_nums1 = re.findall(weight_pattern, n1.lower())
    weight_nums2 = re.findall(weight_pattern, n2.lower())
    
    # Get non-weight numbers
    non_weight_nums1 = [n for n in nums1 if not any(n in w for w in weight_nums1)]
    non_weight_nums2 = [n for n in nums2 if not any(n in w for w in weight_nums2)]
    
    # If there are non-matching, non-weight numbers, products are different
    if non_weight_nums1 and non_weight_nums2 and set(non_weight_nums1) != set(non_weight_nums2):
        logger.info(f"Products have different non-weight numbers: {non_weight_nums1} vs {non_weight_nums2}")
        return False, 0.0
    
    # Clean product names for better comparison
    clean_n1 = preprocess_text(n1)
    clean_n2 = preprocess_text(n2)
    
    # Check for similar structural patterns with different key terms
    words1 = clean_n1.split()
    words2 = clean_n2.split()
    
    # If there's only one differing word and it appears to be a product identifier
    if len(words1) == len(words2) and len(words1) > 2:
        differing_words = []
        for w1, w2 in zip(words1, words2):
            if w1 != w2 and len(w1) > 3 and len(w2) > 3:
                differing_words.append((w1, w2))
        
        # If there's exactly one key difference and the rest of the name is the same,
        # these are likely different products (like different fruits/flavors/etc)
        if len(differing_words) == 1:
            word1, word2 = differing_words[0]
            # Check if these are significantly different words
            word1_emb = get_text_embedding(word1)
            word2_emb = get_text_embedding(word2)
            word_similarity = float(util.cos_sim(word1_emb, word2_emb)[0][0])
            
            # Dynamic threshold based on the similarity of the rest of the words
            common_words1 = [w for w in words1 if w != word1]
            common_words2 = [w for w in words2 if w != word2]
            
            # Calculate similarity of the common words (should be high if structure is the same)
            common_text1 = " ".join(common_words1)
            common_text2 = " ".join(common_words2)
            common_emb1 = get_text_embedding(common_text1)
            common_emb2 = get_text_embedding(common_text2)
            common_similarity = float(util.cos_sim(common_emb1, common_emb2)[0][0])
            
            # If common words are very similar but the differing word is not,
            # these are different products with the same pattern
            # The threshold becomes stricter as the common parts become more similar
            word_similarity_threshold = min(0.8, common_similarity)
            
            # If the differing words are not similar enough relative to context
            if word_similarity < word_similarity_threshold:
                logger.info(f"Products differ by key word: '{word1}' vs '{word2}' with similarity {word_similarity:.2f}")
                return False, t/3.0  # Dynamic score relative to threshold
    
    # Get semantic embeddings for pure comparison
    try:
        # Get embeddings for product names
        e1 = get_text_embedding(clean_n1)
        e2 = get_text_embedding(clean_n2)
        
        # Calculate direct semantic similarity between products
        semantic_similarity = float(util.cos_sim(e1, e2)[0][0] * 100.0)
        
        logger.debug(f"Semantic similarity between '{clean_n1}' and '{clean_n2}': {semantic_similarity:.1f}%")
        
        # Dynamic threshold for semantic similarity based on product name complexity
        # Shorter names need higher similarity to match
        complexity_factor = min(1.0, (len(clean_n1) + len(clean_n2)) / 40.0)
        semantic_threshold = max(35.0, 50.0 - (complexity_factor * 15.0))
        
        # If semantic similarity is below the dynamic threshold
        if semantic_similarity < semantic_threshold:
            logger.info(f"Products seem to be different types: sim={semantic_similarity:.1f}% < threshold={semantic_threshold:.1f}%")
            return False, semantic_similarity
        
    except Exception as e:
        logger.error(f"Error in semantic comparison: {e}")
        semantic_similarity = 0.0
    
    # Get image embeddings if URLs are provided
    i1 = get_image_embedding_from_url(u1) if u1 else None
    i2 = get_image_embedding_from_url(u2) if u2 else None
    
    # Compute all features for matching
    f = compute_features(n1, n2, w1, w2, unit1, unit2, i1, i2)
    
    # Calculate final match score using ML or heuristic
    s = ml_match_score(f)
    
    # Consider products as matching if they meet the threshold
    return (s >= t, round(s, 2))

# ------------------ FAISS INDEX FOR FAST RETRIEVAL ------------------

class FaissIndexWrapper:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.embs = None
        self.ids = []

    def build(self, txts: List[str], ids: List[int]):
        self.ids = ids
        e = []
        for x in txts:
            a = get_text_embedding(preprocess_text(x)).astype(np.float32)
            e.append(a)
        self.embs = np.vstack(e)
        self.index.add(self.embs)

    def search(self, q: str, k: int = 5):
        v = get_text_embedding(preprocess_text(q)).astype(np.float32).reshape(1, -1)
        d, i = self.index.search(v, k)
        r = []
        for dist, idx in zip(d[0], i[0]):
            pid = self.ids[idx]
            r.append((pid, float(dist)))
        return r

# ------------------ FULL PIPELINE ------------------

def run_ai_matching(db_session):
    """Run AI matching on unmatched products with improved connection handling"""
    try:
        logger.info("Starting AI product matching")
        
        # Process in smaller batches to avoid connection timeouts
        BATCH_SIZE = 100
        MAX_RETRIES = 3
        RETRY_DELAY = 5  # seconds
        
        total_matched = 0
        total_created = 0
        total_processed = 0
        
        while True:
            retry_count = 0
            batch_success = False
            
            while not batch_success and retry_count < MAX_RETRIES:
                try:
                    # Get a batch of unmatched products
                    unmatched_products = db_session.query(ProductStoreData).filter(
                        ProductStoreData.product_id.is_(None),
                        ProductStoreData.store_product_name.isnot(None),
                        or_(
                            ProductStoreData.last_matched.is_(None),
                            ProductStoreData.last_matched < func.now() - timedelta(days=7)
                        )
                    ).order_by(func.random()).limit(BATCH_SIZE).all()
                    
                    if not unmatched_products:
                        logger.info("No more unmatched products found")
                        return {
                            "matched": total_matched,
                            "created": total_created,
                            "processed": total_processed
                        }
                    
                    logger.info(f"Processing batch of {len(unmatched_products)} unmatched products")
                    
                    batch_matched = 0
                    batch_created = 0
                    
                    for psd in unmatched_products:
                        try:
                            # Start a nested transaction for each product
                            with db_session.begin_nested():
                                # Skip if no name or skip certain patterns
                                if not psd.store_product_name or "SKIP" in psd.store_product_name:
                                    continue
                                
                                # Find similar products using trigram similarity
                                similar_products = db_session.query(ProductStoreData).filter(
                                    ProductStoreData.store_id != psd.store_id,
                                    func.similarity(ProductStoreData.store_product_name, psd.store_product_name) > 0.3
                                ).order_by(
                                    func.similarity(ProductStoreData.store_product_name, psd.store_product_name).desc()
                                ).limit(20).all()
                                
                                if similar_products:
                                    logger.debug(f"Found {len(similar_products)} potential matches for {psd.store_product_name}")
                                
                                best_match = None
                                best_score = 0.0
                                
                                for candidate in similar_products:
                                    db_session.refresh(candidate)
                                    matched, score = match_products(
                                        psd.store_product_name, candidate.store_product_name,
                                        psd.store_weight_value, candidate.store_weight_value,
                                        psd.store_image_url, candidate.store_image_url,
                                        psd.store_unit, candidate.store_unit,
                                        psd.ean, candidate.ean
                                    )
                                    if matched and score > best_score:
                                        best_match = candidate
                                        best_score = score
                                
                                if best_match and best_score >= 82.0:  # Threshold for matching
                                    logger.info(f"Found AI match for {psd.store_product_name} with score {best_score:.1f}%")
                                    if best_match.product_id:
                                        psd.product_id = best_match.product_id
                                    else:
                                        logger.info(f"Creating new product from AI match for {psd.store_product_name}")
                                        new_product = Product(
                                            name=psd.store_product_name,
                                            weight_value=psd.store_weight_value,
                                            unit_id=psd.store_unit_id,
                                            image_url=psd.store_image_url,
                                            barcode=psd.ean
                                        )
                                        db_session.add(new_product)
                                        db_session.flush()
                                        
                                        psd.product_id = new_product.product_id
                                        best_match.product_id = new_product.product_id
                                        batch_created += 1
                                    
                                    psd.last_matched = func.now()
                                    
                                    match_log = ProductMatchingLog(
                                        product_store_id=psd.product_store_id,
                                        product_id=psd.product_id,
                                        confidence_score=best_score,
                                        matched_by="ai_matcher"
                                    )
                                    db_session.add(match_log)
                                    batch_matched += 1
                                else:
                                    logger.info(f"No match found for {psd.store_product_name}, creating new product")
                                    new_product = Product(
                                        name=psd.store_product_name,
                                        weight_value=psd.store_weight_value,
                                        unit_id=psd.store_unit_id,
                                        image_url=psd.store_image_url,
                                        barcode=psd.ean
                                    )
                                    db_session.add(new_product)
                                    db_session.flush()
                                    
                                    psd.product_id = new_product.product_id
                                    psd.last_matched = func.now()
                                    batch_created += 1
                                    
                                    match_log = ProductMatchingLog(
                                        product_store_id=psd.product_store_id,
                                        product_id=psd.product_id,
                                        confidence_score=0.0,
                                        matched_by="new_product_created"
                                    )
                                    db_session.add(match_log)
                        
                        except Exception as product_error:
                            logger.error(f"Error processing product {psd.store_product_name}: {str(product_error)}")
                            continue
                    
                    # Commit the batch
                    db_session.commit()
                    
                    total_matched += batch_matched
                    total_created += batch_created
                    total_processed += len(unmatched_products)
                    
                    logger.info(f"""Batch completed:
                    - Matched: {batch_matched} products
                    - Created new: {batch_created} products
                    - Total processed: {total_processed}""")
                    
                    batch_success = True
                    
                except Exception as batch_error:
                    retry_count += 1
                    logger.error(f"Batch processing error (attempt {retry_count}/{MAX_RETRIES}): {str(batch_error)}")
                    db_session.rollback()
                    
                    if retry_count < MAX_RETRIES:
                        import time
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.error("Max retries reached, stopping process")
                        return {
                            "matched": total_matched,
                            "created": total_created,
                            "processed": total_processed
                        }
        
    except Exception as e:
        logger.error(f"Error during matching process: {str(e)}", exc_info=True)
        db_session.rollback()
        raise
