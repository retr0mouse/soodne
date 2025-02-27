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

def compare_weight(weight1: float, weight2: float) -> float:
    if not weight1 or not weight2:
        return 0
    try:
        tolerance = 0.05
        return 100 if abs(weight1 - weight2) / max(weight1, weight2) <= tolerance else 0
    except Exception as e:
        logger.error(f"Error comparing weights: {e}")
        return 0

def match_products(product1: models.Product, product2: models.Product, threshold: float = 80.0) -> Tuple[bool, float]:
    name1 = preprocess_name(product1.name)
    name2 = preprocess_name(product2.name)

    name_embedding1 = sentence_model.encode(name1)
    name_embedding2 = sentence_model.encode(name2)
    cosine_sim = np.dot(name_embedding1, name_embedding2) / (
            np.linalg.norm(name_embedding1) * np.linalg.norm(name_embedding2)
    )
    name_similarity = cosine_sim * 100

    fuzzy_similarity = fuzz.token_set_ratio(name1, name2)
    
    combined_name_similarity = (name_similarity * 0.7) + (fuzzy_similarity * 0.3)

    weight_match = compare_weight(product1.weight_value, product2.weight_value)

    image_embedding1 = get_image_embedding(product1.image_url) if product1.image_url else np.zeros(512)
    image_embedding2 = get_image_embedding(product2.image_url) if product2.image_url else np.zeros(512)
    
    if np.linalg.norm(image_embedding1) == 0 or np.linalg.norm(image_embedding2) == 0:
        image_similarity = 0
    else:
        image_cosine_sim = np.dot(image_embedding1, image_embedding2) / (
                np.linalg.norm(image_embedding1) * np.linalg.norm(image_embedding2)
        )
        image_similarity = image_cosine_sim * 100

    total_score = (combined_name_similarity * 0.6) + (weight_match * 0.3) + (image_similarity * 0.1)
    matched = total_score > threshold

    return matched, round(total_score, 2)

def run_ai_matching():
    BATCH_SIZE = 100
    
    with SessionLocal() as db:
        while True:
            unmatched_products = db.query(models.Product).filter(
                models.Product.matching_status == schemas.MatchingStatusEnum.unmatched
            ).limit(BATCH_SIZE).all()
            
            if not unmatched_products:
                break
                
            for product1 in unmatched_products:
                potential_matches = db.query(models.Product).filter(
                    models.Product.matching_status == schemas.MatchingStatusEnum.unmatched,
                    models.Product.product_id != product1.product_id,
                    models.Product.name.ilike(f"%{product1.name[:30]}%")
                ).all()
                
                for product2 in potential_matches:
                    matched, confidence = match_products(product1, product2)
                    if matched:
                        product1.matching_status = schemas.MatchingStatusEnum.matched
                        product2.matching_status = schemas.MatchingStatusEnum.matched
                        log = schemas.ProductMatchingLogCreate(
                            product_id1=product1.id,
                            product_id2=product2.id,
                            confidence_score=confidence,
                            matched_by="ai_matcher"
                        )
                        product_matching_log_service.create(db, log=log)
                        db.commit()
                        break
