from datetime import timedelta
import re
from sqlalchemy import func, or_
from app.core.logger import setup_logger
from app.models.product_store_data import ProductStoreData
from app import schemas
from app.services import product_service
from app.utils.brands import getBrand

logger = setup_logger("app.ai.matcher")


def run_matching(db_session):
    """Run matching based on weight and title similarity"""
    try:
        logger.info("Starting product matching")

        # Get unmatched products (same query as before)
        # unmatched_products = db_session.query(ProductStoreData).filter(
        #     ProductStoreData.product_id.is_(None),
        #     ProductStoreData.store_product_name.isnot(None)
        # ).order_by(func.random()).all()

        # For now let's check all products
        unmatched_products = db_session.query(ProductStoreData).all()

        if not unmatched_products:
            logger.info("No unmatched products found")
            return

        logger.info(f"Found {len(unmatched_products)} unmatched products")

        matched_count = 0
        created_count = 0

        for first_candidate in unmatched_products:
            if not first_candidate.store_product_name or "SKIP" in first_candidate.store_product_name:
                continue

            # Find candidates with similar names using trigram similarity
            similar_products = db_session.query(ProductStoreData).filter(
                ProductStoreData.store_id != first_candidate.store_id,
                func.similarity(ProductStoreData.store_product_name, first_candidate.store_product_name) > 0.3
            ).order_by(
                func.similarity(ProductStoreData.store_product_name, first_candidate.store_product_name).desc()
            ).limit(20).all()

            # Normalize name
            first_candidate_name = first_candidate.store_product_name.lower()
            first_brand = getBrand(first_candidate_name)
            first_candidate_name = re.sub(r"\s+", " ", first_candidate_name)
            first_candidate_name = re.sub(r"[^\w\s]", "", first_candidate_name)
            first_candidate_name = first_candidate_name.replace(first_brand, '') if first_brand else first_candidate_name
            best_match = None
            best_score = 0.0

            for second_candidate in similar_products:
                # Normalize name
                second_candidate_name = second_candidate.store_product_name.lower()
                second_brand = getBrand(second_candidate_name)
                second_candidate_name = re.sub(r"\s+", " ", second_candidate_name)
                second_candidate_name = re.sub(r"[^\w\s]", "", second_candidate_name)
                second_candidate_name = second_candidate_name.replace(second_brand, '') if second_brand else second_candidate_name

                if first_brand != second_brand:
                    continue

                # Convert SQL Decimal to Python float first
                name_similarity = float(
                    db_session.query(
                        func.similarity(first_candidate_name, second_candidate_name)
                    ).scalar()
                )

                # Calculate weight similarity
                if first_candidate.store_weight_value and second_candidate.store_weight_value:
                    weight_diff = abs(float(second_candidate.store_weight_value) - float(first_candidate.store_weight_value))
                    weight_similarity = 1 - (weight_diff / max(float(first_candidate.store_weight_value), 0.0001))
                else:
                    weight_similarity = 0.0

                # Now combine as floats
                total_score = (name_similarity * 0.5 + weight_similarity * 0.5) * 100

                if total_score > best_score:
                    best_match = second_candidate
                    best_score = total_score

            if best_match and best_score >= 75.0:  # Adjusted threshold
                logger.info(f"Match found for {first_candidate.store_product_name} and {best_match.store_product_name} with score {best_score:.1f}%")

                if best_match.product_id is not None:
                    # logger.info(f"Match already has a record in product: {best_match.store_product_name}")
                    first_candidate.product_id = best_match.product_id
                else:
                    # logger.info(f"Creating new product from match for {first_candidate.store_product_name}")

                    new_product = product_service.create(db_session, schemas.ProductCreate(
                        name=first_candidate.store_product_name,
                        weight_value=first_candidate.store_weight_value,
                        unit_id=first_candidate.store_unit_id
                    ))
                    first_candidate.product_id = new_product.product_id
                    created_count += 1

                first_candidate.last_matched = func.now()
                matched_count += 1
                db_session.commit()

            else:
                # logger.info(f"No match found for {first_candidate.store_product_name}, creating new product")
                new_product = product_service.create(db_session, schemas.ProductCreate(
                    name=first_candidate.store_product_name,
                    weight_value=first_candidate.store_weight_value,
                    unit_id=first_candidate.store_unit_id
                ))
                first_candidate.product_id = new_product.product_id
                first_candidate.last_matched = func.now()
                created_count += 1
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

