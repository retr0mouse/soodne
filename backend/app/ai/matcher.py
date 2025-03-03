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

# Initialize both models - general multilingual and Estonian-specific
general_sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
estonian_sentence_model = EstBERTSentenceTransformer()

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
    The function automatically selects the best model without hardcoded language rules.
    """
    if not t:
        # Return zero vector with proper dimensionality for empty text
        return np.zeros(768, dtype=np.float32)  # EstBERT dimension is 768
    
    # For short texts, always use EstBERT to ensure consistency
    if len(t) < 5:
        return estonian_sentence_model.encode(t)
    
    # Get embeddings from both models
    est_embedding = estonian_sentence_model.encode(t)
    gen_embedding = general_sentence_model.encode(t)
    
    # Compute confidence scores for each model by looking at embedding statistics
    # EstBERT will produce more pronounced, confident embeddings for Estonian text
    est_confidence = np.std(est_embedding) * np.max(np.abs(est_embedding))
    gen_confidence = np.std(gen_embedding) * np.max(np.abs(gen_embedding))
    
    # Select the model with higher confidence
    if est_confidence > gen_confidence:
        logger.debug(f"Selected EstBERT model based on embedding confidence")
        return est_embedding
    else:
        logger.debug(f"Selected general model based on embedding confidence")
        return gen_embedding

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
        
        # Get semantic embeddings
        e1 = get_text_embedding(p1)
        e2 = get_text_embedding(p2)
        
        # Calculate cosine similarity for semantic understanding
        semantic_similarity = float(util.cos_sim(e1, e2)[0][0] * 100.0)
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
        # Prioritize semantic understanding
        semantic_weight = 0.85  # Increased semantic understanding weight
        weight_weight = 0.10   # Weight is less important
        image_weight = 0.05    # Image has minimal impact
        
        # Calculate score based primarily on semantic understanding
        score = (f["semantic_similarity"] * semantic_weight + 
                f["weight_similarity"] * weight_weight + 
                f["image_similarity"] * image_weight)
        
        # Boost score for very strong semantic matches
        if f["semantic_similarity"] > 90:
            score += 5.0
        # Penalize low semantic similarities more aggressively
        elif f["semantic_similarity"] < 50:
            score -= 10.0
            
        return max(0, score)  # Ensure score isn't negative
    
    # ML-based scoring if classifier is available
    v = np.array([f["semantic_similarity"], f["weight_similarity"], f["image_similarity"]], 
                 dtype=np.float32).reshape(1, -1)
    p = xgb_clf.predict_proba(v)[0][1]
    return float(p * 100.0)

# ------------------ MATCHING FUNCTION ------------------

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
    
    # Clean product names for better comparison
    clean_n1 = preprocess_text(n1)
    clean_n2 = preprocess_text(n2)
    
    # Get semantic embeddings for pure comparison
    try:
        # Get embeddings for product names
        e1 = get_text_embedding(clean_n1)
        e2 = get_text_embedding(clean_n2)
        
        # Calculate direct semantic similarity between products
        semantic_similarity = float(util.cos_sim(e1, e2)[0][0] * 100.0)
        
        logger.debug(f"Semantic similarity between '{clean_n1}' and '{clean_n2}': {semantic_similarity:.1f}%")
        
        # Determine if products are semantically compatible (same product type)
        # Use pure embedding space distance for compatibility check
        
        # If semantic similarity is very low, products are completely different
        if semantic_similarity < 40:
            logger.info(f"Products seem to be different types: sim={semantic_similarity:.1f}")
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
    """Run AI matching on unmatched products"""
    try:
        logger.info("Starting AI product matching")
        
        # Get all unmatched products with valid names and last_matched older than threshold
        unmatched_products = db_session.query(ProductStoreData).filter(
            ProductStoreData.product_id.is_(None),
            ProductStoreData.store_product_name.isnot(None),
            or_(
                ProductStoreData.last_matched.is_(None),
                ProductStoreData.last_matched < func.now() - timedelta(days=7)
            )
        ).order_by(func.random()).limit(1000).all()
        
        if not unmatched_products:
            logger.info("No unmatched products found")
            return
            
        logger.info(f"Found {len(unmatched_products)} unmatched products")
        
        # Process each unmatched product
        matched_count = 0
        created_count = 0
        
        for psd in unmatched_products:
            # Skip if no name or skip certain patterns
            if not psd.store_product_name or "SKIP" in psd.store_product_name:
                continue
                
            # Find potential semantic matches using embeddings
            embedding = get_text_embedding(preprocess_text(psd.store_product_name))
            
            # Query for similar products using vector similarity (ideally this would use
            # a vector database, but we're simulating the approach)
            similar_products = []
            
            # Fallback to text search if needed
            if not similar_products:
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
                    created_count += 1
                
                psd.last_matched = func.now()
                
                match_log = ProductMatchingLog(
                    product_store_id=psd.product_store_id,
                    product_id=psd.product_id,
                    confidence_score=best_score,
                    matched_by="ai_matcher"
                )
                db_session.add(match_log)
                matched_count += 1
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
                created_count += 1
                
                match_log = ProductMatchingLog(
                    product_store_id=psd.product_store_id,
                    product_id=psd.product_id,
                    confidence_score=0.0,
                    matched_by="new_product_created"
                )
                db_session.add(match_log)
            
            db_session.commit()
            
        logger.info(f"""Matching completed:
        - Matched: {matched_count} products
        - Created new: {created_count} products
        - Success rate: {((matched_count + created_count) / len(unmatched_products) * 100):.1f}%""")
        
        return {
            "matched": matched_count,
            "created": created_count
        }
    except Exception as e:
        logger.error(f"Error during matching process: {str(e)}", exc_info=True)
        db_session.rollback()
        raise
