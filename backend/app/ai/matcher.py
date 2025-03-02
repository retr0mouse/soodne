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
from app.models.unit import Unit
from app.models.enums import MatchingStatusEnum

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    return re.sub(r"[^\w\s]", "", t.lower().strip())

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
    
    def parse(x: str) -> Optional[float]:
        # Remove all non-numeric characters except dots and commas
        c = re.sub(r'[^\d\.,]', '', x)
        # Replace comma with dot for proper float parsing
        c = c.replace(',', '.')
        try:
            return float(c)
        except:
            return None
            
    x = parse(a)
    y = parse(b)
    
    if x is None or y is None:
        return 0.0
        
    # Convert both values to base unit using conversion factors
    if unit_a and unit_b:
        try:
            x = x * float(unit_a.conversion_factor)
            y = y * float(unit_b.conversion_factor)
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error converting units: {e}")
            return 0.0
    
    # Compare with tolerance
    try:
        if x == 0 and y == 0:
            return 100.0
        if x == 0 or y == 0:
            return 0.0
        if abs(x - y)/max(x, y) <= 0.05:
            return 100.0
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
        return heuristic_score(f)
    v = np.array([f["text_cos_sim"], f["weight_sim"], f["image_sim"]], dtype=np.float32).reshape(1, -1)
    p = xgb_clf.predict_proba(v)[0][1]
    return float(p * 100.0)

# ------------------ MATCHING FUNCTION ------------------

def match_products(n1: str, n2: str, w1: Optional[str], w2: Optional[str], 
                   u1: Optional[str], u2: Optional[str],
                   unit1: Optional['Unit'] = None, unit2: Optional['Unit'] = None,
                   e1: Optional[str] = None, e2: Optional[str] = None,
                   t: float = 90.0) -> Tuple[bool, float]:
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
    unmatched_psds = db_session.query(ProductStoreData).filter(
        ProductStoreData.product_id.is_(None),
        ProductStoreData.matching_status == MatchingStatusEnum.unmatched
    ).all()

    known_psds = db_session.query(ProductStoreData).filter(
        ProductStoreData.product_id.isnot(None)
    ).all()

    known_texts = [psd.store_product_name for psd in known_psds]
    known_ids = [psd.product_store_id for psd in known_psds]

    d = len(get_text_embedding("test"))
    faiss_wrapper = FaissIndexWrapper(dim=d)
    faiss_wrapper.build(known_texts, known_ids)

    for psd in unmatched_psds:
        if psd.ean:
            exact_match = db_session.query(ProductStoreData).filter(
                ProductStoreData.store_id != psd.store_id,
                ProductStoreData.product_id.isnot(None),
                ProductStoreData.ean == psd.ean
            ).first()
            if exact_match:
                psd.product_id = exact_match.product_id
                psd.matching_status = MatchingStatusEnum.matched
                psd.last_matched = func.now()
                
                # Log the EAN match
                match_log = ProductMatchingLog(
                    product_store_id=psd.product_store_id,
                    product_id=exact_match.product_id,
                    confidence_score=100.0,
                    matched_by="ean_matcher"
                )
                db_session.add(match_log)
                db_session.commit()
                continue

        candidates = faiss_wrapper.search(psd.store_product_name, k=5)

        best_match_obj = None
        best_score = 0.0

        for cand_id, dist in candidates:
            cand_psd = db_session.query(ProductStoreData).filter(
                ProductStoreData.product_store_id == cand_id
            ).one_or_none()

            if cand_psd:
                matched, score = match_products(
                    psd.store_product_name, cand_psd.store_product_name,
                    psd.store_weight_value, cand_psd.store_weight_value,
                    psd.store_image_url, cand_psd.store_image_url,
                    psd.store_unit, cand_psd.store_unit,
                    psd.ean, cand_psd.ean
                )
                if matched and score > best_score:
                    best_match_obj = cand_psd
                    best_score = score

        # Update product if match found
        if best_match_obj:
            psd.product_id = best_match_obj.product_id
            psd.matching_status = MatchingStatusEnum.matched
            psd.last_matched = func.now()
            
            # Log the match
            match_log = ProductMatchingLog(
                product_store_id=psd.product_store_id,
                product_id=best_match_obj.product_id,
                confidence_score=best_score,
                matched_by="ai_matcher"
            )
            db_session.add(match_log)
        else:
            # No match found - mark as unmatched
            psd.matching_status = MatchingStatusEnum.unmatched
            
            # Log the failed match attempt
            match_log = ProductMatchingLog(
                product_store_id=psd.product_store_id,
                product_id=None,
                confidence_score=0.0,
                matched_by="ai_matcher_no_match"
            )
            db_session.add(match_log)
            
        db_session.commit()
