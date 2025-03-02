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

from app.models.product_store_data import ProductStoreData
from app.models.product_matching_log import ProductMatchingLog
from app.models.product import Product
from app.models.unit import Unit
from app.core.logger import setup_logger

# Replace the existing logger setup
logger = setup_logger("app.ai.matcher")

# ------------------ TEXT & IMAGE MODELS ------------------

sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
    # Convert to lowercase
    text = t.lower().strip()
    
    # Remove specific product variations that shouldn't affect matching
    text = re.sub(r'\s*\d+\s*x\s*', ' ', text)  # Remove quantity indicators like "6x"
    text = re.sub(r'\s*\d+\s*(ml|l|g|kg)\s*', ' ', text, flags=re.IGNORECASE)  # Remove size indicators
    text = re.sub(r'\s*\([^)]*\)\s*', ' ', text)  # Remove parentheses and their contents
    text = re.sub(r'@.*$', '', text)  # Remove @ and everything after it
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Keep some punctuation that might be meaningful
    text = re.sub(r'[^\w\s&\-]', ' ', text)
    
    return text.strip()

def get_text_embedding(t: str) -> np.ndarray:
    return sentence_model.encode(t)

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
    try:
        e1 = get_text_embedding(preprocess_text(n1))
        e2 = get_text_embedding(preprocess_text(n2))
        v = float(util.cos_sim(e1, e2)[0][0] * 100.0)
    except Exception as e:
        logger.error(f"Error computing text similarity: {e}")
        v = 0.0
        
    ws = compare_weight(w1, w2, u1, u2)
    
    iscore = 0.0
    if i1 is not None and i2 is not None:
        try:
            n1_ = np.linalg.norm(i1)
            n2_ = np.linalg.norm(i2)
            if n1_ > 1e-8 and n2_ > 1e-8:
                iscore = float(np.dot(i1, i2) / (n1_ * n2_) * 100.0)
        except Exception as e:
            logger.error(f"Error computing image similarity: {e}")
            
    return {"text_cos_sim": v, "weight_sim": ws, "image_sim": iscore}

def heuristic_score(f: dict) -> float:
    return f["text_cos_sim"] * 0.5 + f["weight_sim"] * 0.25 + f["image_sim"] * 0.25

def ml_match_score(f: dict) -> float:
    if xgb_clf is None:
        # Adjust heuristic weights to give more importance to text similarity
        text_weight = 0.6
        weight_weight = 0.3
        image_weight = 0.1
        
        score = (f["text_cos_sim"] * text_weight + 
                f["weight_sim"] * weight_weight + 
                f["image_sim"] * image_weight)
        
        return score
    
    v = np.array([f["text_cos_sim"], f["weight_sim"], f["image_sim"]], dtype=np.float32).reshape(1, -1)
    p = xgb_clf.predict_proba(v)[0][1]
    return float(p * 100.0)

# ------------------ MATCHING FUNCTION ------------------

def match_products(n1: str, n2: str, w1: Optional[str], w2: Optional[str], 
                   u1: Optional[str], u2: Optional[str],
                   unit1: Optional['Unit'] = None, unit2: Optional['Unit'] = None,
                   e1: Optional[str] = None, e2: Optional[str] = None,
                   t: float = 85.0) -> Tuple[bool, float]:  # Lowered threshold to 85%
    # First check EAN codes if available
    if e1 and e2 and e1 == e2:
        return True, 100.0
        
    i1 = get_image_embedding_from_url(u1) if u1 else None
    i2 = get_image_embedding_from_url(u2) if u2 else None
    f = compute_features(n1, n2, w1, w2, unit1, unit2, i1, i2)
    s = ml_match_score(f)
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
    try:
        # Get all unmatched products in a fresh query
        unmatched_psds = db_session.query(ProductStoreData).filter(
            ProductStoreData.product_id.is_(None)
        ).all()

        total_products = len(unmatched_psds)
        if not unmatched_psds:
            logger.info("No unmatched products found to process")
            return {
                "processed": 0,
                "matched": 0,
                "created": 0
            }

        logger.info(f"Starting matching process for {total_products} unmatched products")
        
        matched_count = 0
        created_count = 0
        processed_count = 0
        
        for psd in unmatched_psds:
            try:
                processed_count += 1
                if processed_count % 100 == 0:  # Log progress every 100 items
                    logger.info(f"Progress: {processed_count}/{total_products} ({(processed_count/total_products)*100:.1f}%)")
                
                # Refresh the object to ensure it's bound to the session
                db_session.refresh(psd)
                
                logger.debug(f"Processing product: {psd.store_product_name} (ID: {psd.product_store_id})")
                
                # First try to find by EAN
                if psd.ean:
                    logger.debug(f"Trying EAN match for {psd.ean}")
                    exact_match = db_session.query(ProductStoreData).filter(
                        ProductStoreData.store_id != psd.store_id,
                        ProductStoreData.ean == psd.ean
                    ).first()
                    
                    if exact_match:
                        db_session.refresh(exact_match)
                        if exact_match.product_id:
                            logger.info(f"Found exact EAN match for {psd.store_product_name} with existing product")
                            psd.product_id = exact_match.product_id
                        else:
                            logger.info(f"Creating new product from EAN match for {psd.store_product_name}")
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
                            exact_match.product_id = new_product.product_id
                            created_count += 1
                        
                        psd.last_matched = func.now()
                        
                        match_log = ProductMatchingLog(
                            product_store_id=psd.product_store_id,
                            product_id=psd.product_id,
                            confidence_score=100.0,
                            matched_by="ean_matcher"
                        )
                        db_session.add(match_log)
                        matched_count += 1
                        db_session.commit()
                        continue

                # Try to find similar products
                logger.debug(f"Searching for similar products to {psd.store_product_name}")
                similar_products = db_session.query(ProductStoreData).filter(
                    ProductStoreData.store_id != psd.store_id,
                    ProductStoreData.store_product_name.ilike(f"%{psd.store_product_name}%")
                ).all()

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

                if best_match and best_score >= 85.0:  # Threshold for matching
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
                
            except Exception as e:
                logger.error(f"Error processing product {psd.store_product_name}: {str(e)}", exc_info=True)
                db_session.rollback()
                continue

        logger.info(f"""Matching completed:
        - Processed: {processed_count} products
        - Matched: {matched_count} products
        - Created new: {created_count} products
        - Success rate: {((matched_count + created_count) / processed_count * 100):.1f}%""")
        
        return {
            "processed": processed_count,
            "matched": matched_count,
            "created": created_count
        }
    except Exception as e:
        logger.error(f"Error during matching process: {str(e)}", exc_info=True)
        db_session.rollback()
        raise
