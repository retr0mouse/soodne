import logging
from typing import Tuple
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import models as torchvision_models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from rapidfuzz import fuzz
import requests
from io import BytesIO
import re

from app import models, schemas
from app.database.database import SessionLocal
from app.services import product_matching_log_service

logger = logging.getLogger(__name__)

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
image_model = torchvision_models.resnet18(weights=ResNet18_Weights.DEFAULT)
image_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_name(name: str) -> str:
    name = re.sub(r'[^\w\s]', '', name)
    name = name.lower()
    return name

def get_image_embedding(url: str) -> np.ndarray:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            embedding = image_model(image).numpy().flatten()
        return embedding
    except Exception as e:
        logger.error(f"Error processing image from URL {url}: {e}")
        return np.zeros(512)

def compare_weight(psd1: models.ProductStoreData, psd2: models.ProductStoreData) -> float:
    if not psd1.store_weight_value or not psd2.store_weight_value:
        return 0
    try:
        weight1 = float(psd1.store_weight_value)
        weight2 = float(psd2.store_weight_value)
        tolerance = 0.05
        return 100 if abs(weight1 - weight2) / max(weight1, weight2) <= tolerance else 0
    except Exception as e:
        logger.error(f"Error comparing weights: {e}")
        return 0

def match_products(psd1: models.ProductStoreData, psd2: models.ProductStoreData) -> Tuple[bool, float]:
    name1 = preprocess_name(psd1.store_product_name)
    name2 = preprocess_name(psd2.store_product_name)

    name_embedding1 = sentence_model.encode(name1)
    name_embedding2 = sentence_model.encode(name2)
    cosine_sim = np.dot(name_embedding1, name_embedding2) / (
            np.linalg.norm(name_embedding1) * np.linalg.norm(name_embedding2)
    )
    name_similarity = cosine_sim * 100

    fuzzy_similarity = fuzz.token_set_ratio(name1, name2)

    combined_name_similarity = (name_similarity * 0.7) + (fuzzy_similarity * 0.3)

    weight_match = compare_weight(psd1, psd2)

    total_score = (combined_name_similarity * 0.7) + (weight_match * 0.3)
    matched = total_score > 80

    return matched, round(total_score, 2)

def run_ai_matching():
    with SessionLocal() as db:
        unmatched_psds = db.query(models.ProductStoreData).filter(
            models.ProductStoreData.matching_status == schemas.MatchingStatusEnum.unmatched
        ).all()
        for psd in unmatched_psds:
            similar_psds = db.query(models.ProductStoreData).filter(
                models.ProductStoreData.store_id != psd.store_id,
                models.ProductStoreData.matching_status == schemas.MatchingStatusEnum.unmatched
            ).all()
            for similar_psd in similar_psds:
                matched, confidence = match_products(psd, similar_psd)
                if matched:
                    psd.matching_status = schemas.MatchingStatusEnum.matched
                    similar_psd.matching_status = schemas.MatchingStatusEnum.matched
                    log = schemas.ProductMatchingLogCreate(
                        product_store_id=psd.product_store_id,
                        product_id=similar_psd.product_id,
                        confidence_score=confidence,
                        matched_by="ai_matcher"
                    )
                    product_matching_log_service.create(db, log=log)
                    db.commit()
                    break
