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
from sqlalchemy import func

from app import models, schemas
from app.database.database import SessionLocal
from app.services import product_matching_log_service, product_service

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
    # If EAN codes are available and match, it's a definite match
    if psd1.ean and psd2.ean and psd1.ean == psd2.ean:
        return True, 100.0

    # Compare units and weights if available
    weight_match = compare_weight(psd1, psd2)
    if psd1.store_unit_id and psd2.store_unit_id and psd1.store_unit_id != psd2.store_unit_id:
        return False, 0.0

    # Get base product names by removing common descriptors
    name1 = psd1.store_product_name.lower()
    name2 = psd2.store_product_name.lower()
    
    # Use ML model to get semantic embeddings
    emb1 = sentence_model.encode(name1)
    emb2 = sentence_model.encode(name2)
    semantic_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))) * 100

    # If semantic similarity is too low, they're different products
    if semantic_sim < 50:  # Products are semantically very different
        return False, 0.0

    # Calculate trigram similarity using database
    with SessionLocal() as db:
        trigram_sim = db.execute(
            """
            SELECT similarity(LOWER(:name1), LOWER(:name2)) * 100 as sim
            """,
            {"name1": name1, "name2": name2}
        ).scalar()

    # Get token sort ratio for more precise comparison
    token_sim = fuzz.token_sort_ratio(name1, name2)
    
    # Extract main product type (last word before unit)
    def get_product_type(name: str) -> str:
        # Remove unit if present
        name = re.sub(r'\s*,\s*kg$|\s*kg$', '', name)
        # Get last word before any descriptors
        words = name.split()
        if len(words) > 0:
            return words[-1]
        return name

    prod_type1 = get_product_type(name1)
    prod_type2 = get_product_type(name2)

    # If main product types are different, they're different products
    if prod_type1 != prod_type2:
        type_sim = fuzz.ratio(prod_type1, prod_type2)
        if type_sim < 80:  # Allow some fuzzy matching for typos
            return False, 0.0

    # Calculate final similarity score with weighted components
    name_similarity = (
        semantic_sim * 0.4 +
        trigram_sim * 0.3 +     
        token_sim * 0.3           
    )

    # Final score combining name and weight matching
    total_score = (name_similarity * 0.8) + (weight_match * 0.2)

    # Require very high confidence for matching
    matched = total_score > 95

    return matched, round(total_score, 2)

def run_ai_matching():
    with SessionLocal() as db:
        # Get all unmatched product store data entries
        unmatched_psds = db.query(models.ProductStoreData).filter(
            models.ProductStoreData.product_id.is_(None),
            models.ProductStoreData.matching_status == schemas.MatchingStatusEnum.unmatched
        ).all()

        for psd in unmatched_psds:
            # First try to find exact EAN matches
            if psd.ean:
                exact_match = db.query(models.ProductStoreData).filter(
                    models.ProductStoreData.store_id != psd.store_id,
                    models.ProductStoreData.product_id.isnot(None),
                    models.ProductStoreData.ean == psd.ean
                ).first()
                
                if exact_match:
                    psd.product_id = exact_match.product_id
                    psd.matching_status = schemas.MatchingStatusEnum.matched
                    psd.last_matched = func.now()
                    
                    log = schemas.ProductMatchingLogCreate(
                        product_store_id=psd.product_store_id,
                        product_id=exact_match.product_id,
                        confidence_score=100.0,
                        matched_by="ean_matcher"
                    )
                    product_matching_log_service.create(db, log=log)
                    db.commit()
                    continue

            # Find potential matches using both trigram and semantic similarity
            store_name = psd.store_product_name.lower()
            
            # Get base product type
            product_type = store_name.split()[-1].strip(',.') if store_name else ''
            
            # Find potential matches with similar names
            potential_matches = db.execute(
                """
                SELECT psd.* 
                FROM product_store_data psd
                WHERE psd.store_id != :store_id
                AND psd.product_id IS NOT NULL
                AND (
                    similarity(LOWER(psd.store_product_name), LOWER(:name)) > 0.3
                    OR LOWER(psd.store_product_name) LIKE '%' || :product_type || '%'
                )
                ORDER BY similarity(LOWER(psd.store_product_name), LOWER(:name)) DESC
                LIMIT 5
                """,
                {
                    "store_id": psd.store_id,
                    "name": store_name,
                    "product_type": product_type
                }
            ).fetchall()

            best_match = None
            best_confidence = 0

            for potential_match in potential_matches:
                matched, confidence = match_products(psd, potential_match)
                if matched and confidence > best_confidence:
                    best_match = potential_match
                    best_confidence = confidence

            if best_match:
                psd.product_id = best_match.product_id
                psd.matching_status = schemas.MatchingStatusEnum.matched
                psd.last_matched = func.now()

                log = schemas.ProductMatchingLogCreate(
                    product_store_id=psd.product_store_id,
                    product_id=best_match.product_id,
                    confidence_score=best_confidence,
                    matched_by="hybrid_matcher"
                )
                product_matching_log_service.create(db, log=log)
            else:
                # No match found - create new product
                new_product = schemas.ProductCreate(
                    name=psd.store_product_name,
                    image_url=psd.store_image_url,
                    weight_value=psd.store_weight_value,
                    unit_id=psd.store_unit_id
                )
                created_product = product_service.create(db, product=new_product)

                psd.product_id = created_product.product_id
                psd.matching_status = schemas.MatchingStatusEnum.matched
                psd.last_matched = func.now()

                log = schemas.ProductMatchingLogCreate(
                    product_store_id=psd.product_store_id,
                    product_id=created_product.product_id,
                    confidence_score=100.0,
                    matched_by="new_product_creation"
                )
                product_matching_log_service.create(db, log=log)

            db.commit()
