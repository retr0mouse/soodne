from datetime import timedelta, datetime
import re
from sqlalchemy import func
from app.core.logger import setup_logger
from app.models.product_store_data import ProductStoreData
from app.models.unit import Unit
from app import schemas
from app.services import product_service
from app.utils.brands import getBrand
from collections import Counter
import string
import os
import pandas as pd
from rapidfuzz import fuzz
from typing import Dict, List, Tuple, Optional, Set, Union, Any

logger = setup_logger("app.ai.matcher")

def normalize_string(text: Optional[str]) -> Optional[str]:
    """Normalize text by converting to lowercase, removing extra spaces and special characters, and sort words"""
    if not text:
        return text
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Sort words
    words = text.split()
    words.sort()
    return ' '.join(words)

class EstonianProductNLP:
    def __init__(self):
        """Initialize the class with additional attributes"""
        # Base initialization
        self.common_words: Set[str] = set()
        self.taste_markers: List[str] = []
        self.units_cache: Dict[int, float] = {}
        self.dynamic_common_words: Dict[str, float] = {}
        self.common_words_threshold: float = 0.2
        self.min_word_length: int = 2
        self.dynamic_taste_markers: Dict[str, int] = {}
        self.dynamic_taste_words: Dict[str, int] = {}
        
        # Regular expression for weights
        self.weight_pattern = r'\b\d+x\d+[a-z]*\b|\b\d+[a-z]*\b'
        
        # Initialize dictionaries for search optimization
        self.stop_words = {'ja', 'ning', 'kui', 'või', 'aga', 'vaid', 'ka', 'ei', 'et', 'see', 'ning'}
        # Common abbreviations and their full forms
        self.abbreviations = {
            'g': 'gramm',
            'kg': 'kilogramm',
            'l': 'liiter',
            'ml': 'milliliiter',
            'tk': 'tükk',
            'vp': 'vaakumpakendis',
        }
        
    def initialize_from_data(self, db_session):
        """Initializes all dynamic lists from database data"""
        units = db_session.query(Unit).all()
        for unit in units:
            self.units_cache[unit.unit_id] = unit.conversion_factor
            
        product_data = db_session.query(ProductStoreData).all()
        product_names = [p.store_product_name for p in product_data if p.store_product_name]
        
        base_taste_markers = ['maitsega', 'maitse', 'mts', 'maits', 'ga', 'ega', 'maitseline', 
                              'maitsest', 'maits-', '-maits', 'maitsestatud', 'maustatud', 
                              'aroom', 'lõhn', 'maitseline', 'maitsel', 'mait.', 'm-ga']
        for marker in base_taste_markers:
            self.taste_markers.append(marker)
        
        self.update_common_words(product_names)
        self.update_taste_markers(product_names)
        
    def update_taste_markers(self, product_names):
        """Dynamically identifies taste markers based on product analysis"""
        if not product_names:
            return
            
        for name in product_names:
            if not name:
                continue
                
            # Find flavor patterns in Estonian
            flavor_et = re.findall(r'([a-zA-ZäöüõÄÖÜÕ]+(?:\s+[a-zA-ZäöüõÄÖÜÕ]+)?)\s+maitse[a-zA-ZäöüõÄÖÜÕ]*', name.lower())
            for match in flavor_et:
                self.dynamic_taste_words[match.strip()] = self.dynamic_taste_words.get(match.strip(), 0) + 1
            
            for marker in self.taste_markers:
                if marker in name.lower():
                    words = name.lower().split()
                    marker_index = -1
                    for i, word in enumerate(words):
                        if marker in word:
                            marker_index = i
                            break
                    
                    if marker_index >= 0 and marker_index < len(words) - 1:
                        next_word = words[marker_index + 1]
                        if len(next_word) > 2:
                            self.dynamic_taste_words[next_word] = self.dynamic_taste_words.get(next_word, 0) + 1
        
        total_products = len(product_names)
        threshold = 0.01
        
        for word, count in self.dynamic_taste_words.items():
            if count / total_products >= threshold and word not in self.taste_markers:
                suffix_markers = [word + s for s in ['ga', 'ega', 'ne', 'line', 'st']]
                self.taste_markers.extend(suffix_markers)
        
        for name in product_names:
            if not name:
                continue
                
            words = name.lower().split()
            for i, word in enumerate(words):
                if i > 0 and word in self.dynamic_taste_words and self.dynamic_taste_words[word] > 5:
                    prev_word = words[i-1]
                    if len(prev_word) > 2 and prev_word not in self.taste_markers:
                        self.dynamic_taste_markers[prev_word] = self.dynamic_taste_markers.get(prev_word, 0) + 1
        
        for marker, count in self.dynamic_taste_markers.items():
            if count > 3 and marker not in self.taste_markers:
                self.taste_markers.append(marker)
        
        logger.debug(f"Updated taste markers list. Total markers: {len(self.taste_markers)}")
        
    def update_common_words(self, product_names, force_threshold=None):
        """Dynamically updates the list of common words based on analysis of existing product names"""
        if not product_names:
            return
            
        all_words = []
        for name in product_names:
            if name:
                tokens = self._tokenize(name.lower())
                all_words.extend([t for t in tokens if len(t) > self.min_word_length])
                
        word_counts = Counter(all_words)
        total_products = len(product_names)
        
        for word, count in word_counts.items():
            self.dynamic_common_words[word] = count / total_products
            
        threshold = force_threshold if force_threshold is not None else self.common_words_threshold
        freq_common_words = {word for word, freq in self.dynamic_common_words.items() 
                           if freq >= threshold}
        
        self.common_words.update(freq_common_words)
        
        logger.debug(f"Updated common words list. Total words: {len(self.common_words)}")
        
    def extract_features(self, text):
        """Extract features from product text"""
        if not text:
            return [], []
            
        normalized_text = normalize_string(text)
        
        # Split text into tokens
        tokens = self._tokenize(normalized_text)
        
        # Extract attributes (words in brackets, after commas, etc.)
        attributes = []
        
        # Find words in brackets
        bracket_pattern = re.compile(r'\((.*?)\)')
        bracket_matches = bracket_pattern.findall(normalized_text)
        for match in bracket_matches:
            # Add string values, not tuples
            attributes.extend([token for token in self._tokenize(match)])
            normalized_text = bracket_pattern.sub('', normalized_text)
        
        # Find parts after comma
        comma_parts = normalized_text.split(',')
        if len(comma_parts) > 1:
            normalized_text = comma_parts[0]
            for part in comma_parts[1:]:
                # Make sure we're adding strings
                attributes.extend([token for token in self._tokenize(part)])
        
        # Define the pattern for weights directly
        weight_pattern = r'\b\d+x\d+[a-z]*\b|\b\d+[a-z]*\b'
        
        # Find weights and units of measurement
        weight_matches = re.findall(weight_pattern, normalized_text)
        for match in weight_matches:
            if match and match not in attributes:
                attributes.append(match)
        
        # Get the main words
        main_words = self._tokenize(normalized_text)
        
        # Remove duplicates and ensure all elements are strings
        main_words = [str(word) for word in list(set(main_words))]
        attributes = [str(attr) for attr in list(set(attributes))]
        
        return main_words, attributes
    
    def _tokenize(self, text):
        """Tokenize and sort words"""
        text = normalize_string(text)
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        tokens = [word.strip() for word in text.split() if word.strip()]
        tokens.sort()  # Sort tokens
        return tokens
    
    def _is_common_word(self, word):
        """Check if the word is a common word"""
        # Normalize the word before checking
        word = normalize_string(word.lower())
        
        return (word in self.common_words or 
                word in self.dynamic_common_words and 
                self.dynamic_common_words[word] >= self.common_words_threshold)

    def _improved_word_similarity(self, words1, words2):
        """
        Improved comparison of word sets taking into account partial matches,
        word normalization, and mandatory presence of all critical words.
        """
        if not words1 or not words2:
            return 0.0

        # Normalize words and process abbreviations
        norm_words1 = []
        for w in words1:
            w = w.rstrip('.')
            norm_words1.append(normalize_string(w))
        
        norm_words2 = []
        for w in words2:
            w = w.rstrip('.')
            norm_words2.append(normalize_string(w))
        
        # Find critical words (words longer than 3 characters that may indicate
        # product characteristics - flavors, models, etc.)
        critical_words1 = [w for w in norm_words1 if len(w) > 3 and not self._is_common_word(w)]
        critical_words2 = [w for w in norm_words2 if len(w) > 3 and not self._is_common_word(w)]
        
        # Non-critical words
        non_critical_words1 = [w for w in norm_words1 if w not in critical_words1 and len(w) > 2]
        non_critical_words2 = [w for w in norm_words2 if w not in critical_words2 and len(w) > 2]
        
        # Check each critical word from the first set
        for word1 in critical_words1:
            best_match_score = 0
            for word2 in critical_words2:
                # Check for abbreviations
                if word1.startswith(word2) and len(word2) >= 3:
                    # If word2 is the beginning of word1 and long enough
                    similarity = 0.9
                elif word2.startswith(word1) and len(word1) >= 3:
                    # If word1 is the beginning of word2 and long enough
                    similarity = 0.9
                # Calculate word similarity
                elif word1 == word2:
                    similarity = 1.0
                elif word1 in word2 or word2 in word1:
                    # One word is fully contained in another
                    similarity = 0.9
                else:
                    similarity = self._levenshtein_similarity(word1, word2)
                
                best_match_score = max(best_match_score, similarity)
            
            # Critical words need 80% match
            if best_match_score < 0.8:
                return 0.0
        
        # Same for the second set of critical words
        for word2 in critical_words2:
            best_match_score = 0
            for word1 in critical_words1:
                if word2 == word1:
                    similarity = 1.0
                elif word2 in word1 or word1 in word2:
                    similarity = 0.9
                else:
                    similarity = self._levenshtein_similarity(word2, word1)
                
                best_match_score = max(best_match_score, similarity)
            
            if best_match_score < 0.8:
                return 0.0
        
        # Check non-critical words with lower threshold (65%)
        non_critical_mismatch_count = 0
        for word1 in non_critical_words1:
            best_match_score = 0
            for word2 in non_critical_words2:
                # Calculate similarity as before
                if word1 == word2:
                    similarity = 1.0
                elif word1 in word2 or word2 in word1:
                    similarity = 0.9
                else:
                    similarity = self._levenshtein_similarity(word1, word2)
                
                best_match_score = max(best_match_score, similarity)
            
            # For non-critical words, use lower threshold
            if best_match_score < 0.70:
                non_critical_mismatch_count += 1
        
        # Penalize if too many non-critical words don't match well
        non_critical_mismatch_ratio = non_critical_mismatch_count / len(non_critical_words1) if non_critical_words1 else 0
        
        # Original code for calculating overall similarity, adjusted for non-critical mismatches
        total_similarity = 0.0
        max_possible = max(len(norm_words1), len(norm_words2))
        
        if max_possible == 0:
            return 0.0
        
        # Count total matching words
        matches = 0
        for word1 in norm_words1:
            for word2 in norm_words2:
                if word1 == word2 or word1 in word2 or word2 in word1 or self._levenshtein_similarity(word1, word2) > 0.8:
                    matches += 1
                    break
        
        # Reduce final score based on non-critical mismatches
        similarity_score = matches / max_possible
        if non_critical_mismatch_ratio > 0.5:  # If more than half of non-critical words don't match well
            similarity_score *= (1 - 0.2 * non_critical_mismatch_ratio)  # Reduce score by up to 20%
        
        return similarity_score

    def _levenshtein_similarity(self, str1, str2):
        """
        Calculate similarity measure using the rapidfuzz library.
        Returns a value from 0 to 1, where 1 is a perfect match.
        """
        if not str1 or not str2:
            return 0.0
        
        # Normalize strings before comparison
        str1 = normalize_string(str1.lower())
        str2 = normalize_string(str2.lower())
        
        # If strings are identical, return 1 immediately
        if str1 == str2:
            return 1.0
        
        # Use fuzz.ratio from rapidfuzz to get string similarity
        similarity = fuzz.ratio(str1, str2) / 100.0
        
        return similarity

    def _attribute_similarity(self, attributes1, attributes2):
        """
        Calculate the similarity of two attribute sets.
        Takes into account both exact and partial matches.
        """
        if not attributes1 or not attributes2:
            # If one of the sets is empty, return 0 or a small value
            # If both sets are empty, this can be considered a good match
            if not attributes1 and not attributes2:
                return 1.0
            return 0.1  # Small value to not completely discard products
        
        # Normalize attributes
        norm_attrs1 = []
        for attr in attributes1:
            try:
                # Convert to string if it's not a string
                if not isinstance(attr, str):
                    attr = str(attr)
                norm_attrs1.append(normalize_string(attr))
            except:
                continue
            
        norm_attrs2 = []
        for attr in attributes2:
            try:
                # Convert to string if it's not a string
                if not isinstance(attr, str):
                    attr = str(attr)
                norm_attrs2.append(normalize_string(attr))
            except:
                continue
        
        # If sets are empty after normalization
        if not norm_attrs1 or not norm_attrs2:
            if not norm_attrs1 and not norm_attrs2:
                return 1.0
            return 0.1
        
        # Find exact matches
        exact_matches = set(norm_attrs1) & set(norm_attrs2)
        exact_match_count = len(exact_matches)
        
        # Look for partial matches for attributes that did not match exactly
        partial_matches_score = 0.0
        remaining_attrs1 = [a for a in norm_attrs1 if a not in exact_matches]
        remaining_attrs2 = [a for a in norm_attrs2 if a not in exact_matches]
        
        for attr1 in remaining_attrs1:
            best_partial_match = 0.0
            for attr2 in remaining_attrs2:
                # Skip attributes that are too short
                if len(attr1) < 2 or len(attr2) < 2:
                    continue
                    
                # Use Levenshtein distance to determine similarity
                similarity = self._levenshtein_similarity(attr1, attr2)
                    
                if similarity > 0.8:  # Threshold for partial match
                    best_partial_match = max(best_partial_match, similarity * 0.8)
            
            partial_matches_score += best_partial_match
        
        # Calculate final similarity score
        total_possible = max(len(norm_attrs1), len(norm_attrs2))
        if total_possible == 0:
            return 1.0  # Both sets are empty after filtering
        
        # Combine exact and partial matches
        similarity_score = (exact_match_count + partial_matches_score) / total_possible
        
        return similarity_score

    def _simple_unique_words(self, product1, product2):
        """
        Compare unique words in product names.
        Ignores common words and focuses on specific words.
        """
        if not product1 or not product2:
            return 0.0
        
        # Normalize and split into words
        words1 = [w.rstrip('.') for w in normalize_string(product1.lower()).split()]
        words2 = [w.rstrip('.') for w in normalize_string(product2.lower()).split()]
        
        # Filter out common words and too short words
        unique_words1 = [w for w in words1 if len(w) > 3 and not self._is_common_word(w)]
        unique_words2 = [w for w in words2 if len(w) > 3 and not self._is_common_word(w)]
        
        # If there are no unique words, consider products similar
        if not unique_words1 and not unique_words2:
            return 1.0
        elif not unique_words1 or not unique_words2:
            return 0.0
        
        # Find matching unique words
        matches = 0
        for word1 in unique_words1:
            for word2 in unique_words2:
                # Normalize before comparison
                norm_word1 = normalize_string(word1)
                norm_word2 = normalize_string(word2)
                
                # Check for exact match or one word contained in another
                if norm_word1 == norm_word2 or (len(norm_word1) > 4 and norm_word1 in norm_word2) or (len(norm_word2) > 4 and norm_word2 in norm_word1):
                    matches += 1
                    break
        
        # Calculate match score
        max_unique = max(len(unique_words1), len(unique_words2))
        if max_unique > 0:
            return matches / max_unique
        
        return 0.5  # Return median value if there are no unique words

    def _compare_weights(self, product1, product2, units_dict):
        """
        Compare weights of two products taking into account units of measurement.
        """
        # Dictionary of multipliers and their regular expression for Estonian
        unit_patterns = {
            'g': 1.0,
            'gr': 1.0,
            'gr.': 1.0,
            'kg': 1000.0,
            'kg.': 1000.0,
            'l': 1.0,
            'l.': 1.0,
            'ml': 0.001,
            'ml.': 0.001
        }
        
        # Use units_dict if provided, otherwise use standard patterns
        patterns_to_use = units_dict if units_dict else unit_patterns
        
        # Create regular expression to find weights
        units_pattern = '|'.join(patterns_to_use.keys())
        weight_pattern = rf'(\d+[.,]?\d*)\s*({units_pattern})'
        
        weight1 = re.findall(weight_pattern, product1.lower().replace(',', '.'))
        weight2 = re.findall(weight_pattern, product2.lower().replace(',', '.'))
        
        # If either product has no weight information, consider weights matching
        if not weight1 or not weight2:
            return True
        
        # Get value and unit for first product
        value1 = float(weight1[0][0])
        unit1 = weight1[0][1]
        
        # Get value and unit for second product
        value2 = float(weight2[0][0])
        unit2 = weight2[0][1]
        
        # Convert to base units
        base_value1 = value1 * patterns_to_use.get(unit1, 1.0)
        base_value2 = value2 * patterns_to_use.get(unit2, 1.0)
        
        # Compare with small margin of error
        return abs(base_value1 - base_value2) < 0.01

    def compare_products(self, product1, product2, units_dict=None):
        """Compare two products with improved logic for handling critical differences"""
        main_words1, attributes1 = self.extract_features(product1)
        main_words2, attributes2 = self.extract_features(product2)
        
        # Check weight/volume of the product considering units
        if not self._compare_weights(product1, product2, units_dict or {}):
            return 0.0
        
        # Compare main words taking critical attributes into account
        word_similarity = self._improved_word_similarity(main_words1, main_words2)
        
        # If main words don't match well enough, stop comparison
        if word_similarity < 0.3:
            return 0.0
        
        # Compare attributes
        attribute_score = self._attribute_similarity(attributes1, attributes2)
        
        # Compare unique words
        unique_words_score = self._simple_unique_words(product1, product2)
        
        # Updated weight distribution
        total_score = (
            word_similarity * 0.4 + 
            attribute_score * 0.4 + 
            unique_words_score * 0.2
        )
        
        # Return percentage for better readability
        return total_score * 100

estonian_nlp = EstonianProductNLP()

# Function to save matches to Excel file
def save_potential_matches_to_file(first_product, second_product, score, metrics):
    """Saves information about potential matches to an Excel file"""
    try:
        filepath = "potential_matches.xlsx"
        
        # Create data for writing
        data = {
            'First Product': [first_product],
            'Second Product': [second_product],
            'Score': [f"{score:.1f}%"],
            'Word Similarity': [f"{metrics.get('word_similarity', 0):.1f}%"],
            'Attribute Score': [f"{metrics.get('attribute_score', 0):.1f}%"],
            'Unique Words Score': [f"{metrics.get('unique_words_score', 0):.1f}%"],
            'First Main Words': [str(metrics.get('first_main_words', []))],
            'Second Main Words': [str(metrics.get('second_main_words', []))],
            'First Attributes': [str(metrics.get('first_attributes', []))],
            'Second Attributes': [str(metrics.get('second_attributes', []))],
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        
        # Create DataFrame from data
        df = pd.DataFrame(data)
        
        sheet_name = "Matches"
        
        # If file exists, load it and add new data
        if os.path.isfile(filepath):
            try:
                # Check if "Matches" sheet exists in the file
                try:
                    existing_df = pd.read_excel(filepath, sheet_name=sheet_name)
                    # If sheet exists, add data
                    with pd.ExcelWriter(filepath, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                        df.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1, sheet_name=sheet_name)
                except ValueError as ve:
                    # If "Matches" sheet is not found, create a new sheet
                    with pd.ExcelWriter(filepath, mode='a', engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name=sheet_name)
            except Exception as e:
                logger.error(f"Error adding to existing Excel file: {str(e)}")
                # If failed to add to existing file, overwrite the file
                df.to_excel(filepath, index=False, engine='openpyxl', sheet_name=sheet_name)
        else:
            # If file doesn't exist, create a new one
            df.to_excel(filepath, index=False, engine='openpyxl', sheet_name=sheet_name)
        
        logger.debug(f"Match data successfully saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving matches to Excel file: {str(e)}")

def run_matching(db_session):
    """Run matching based on weight and title similarity with Estonian NLP"""
    try:
        logger.info("Starting product matching with Estonian NLP")

        estonian_nlp.initialize_from_data(db_session)

        # Common function to create a new product
        def create_new_product(store_product):
            new_product = product_service.create(db_session, schemas.ProductCreate(
                name=store_product.store_product_name,
                weight_value=store_product.store_weight_value,
                unit_id=store_product.store_unit_id,
                barcode=getattr(store_product, 'ean', None)
            ))
            store_product.product_id = new_product.product_id
            store_product.last_matched = func.now()
            return new_product
        
        unmatched_products = db_session.query(ProductStoreData).all()
        
        ean_matched_count = 0
        for first_candidate in unmatched_products:
            if not first_candidate.ean:
                continue
                
            ean_matches = db_session.query(ProductStoreData).filter(
                ProductStoreData.store_id != first_candidate.store_id,
                ProductStoreData.ean == first_candidate.ean,
                ProductStoreData.ean.isnot(None)
            ).all()
            
            if ean_matches:
                ean_match = ean_matches[0]
                
                if ean_match.product_id is not None:
                    # Check that this product_id is not already used for another product
                    existing_product = db_session.query(schemas.Product).filter(
                        schemas.Product.product_id == ean_match.product_id
                    ).first()
                    
                    if existing_product:
                        first_candidate.product_id = ean_match.product_id
                        first_candidate.last_matched = func.now()
                        ean_matched_count += 1
                        db_session.commit()
                        
                        if not existing_product.barcode:
                            existing_product.barcode = first_candidate.ean
                            db_session.commit()
                
                elif first_candidate.product_id is not None:
                    for match in ean_matches:
                        match.product_id = first_candidate.product_id
                        match.last_matched = func.now()
                    ean_matched_count += len(ean_matches)
                    db_session.commit()
                else:
                    new_product = create_new_product(first_candidate)
                    ean_matched_count += 1
                    db_session.commit()
        
        logger.info(f"EAN matching completed: found {ean_matched_count} matches")
        
        # Filter products that already have product_id
        unmatched_products = [p for p in unmatched_products if p.product_id is None]
        
        category_index = {}
        for product in unmatched_products:
            category_key = getattr(product, "category_id", None) 
            if category_key is None:
                category_key = "default"
                
            if category_key not in category_index:
                category_index[category_key] = []
            category_index[category_key].append(product)

        if not unmatched_products:
            logger.info("No unmatched products found")
            return

        logger.info(f"Found {len(unmatched_products)} unmatched products in {len(category_index)} categories for NLP analysis")

        matched_count = 0
        created_count = 0

        category_thresholds = {}
        brand_thresholds = {}
        
        existing_products = db_session.query(ProductStoreData).filter(ProductStoreData.product_id.isnot(None)).all()
        for product in existing_products:
            product_category = getattr(product, "category_id", None)
            if product_category:
                if product_category not in category_thresholds:
                    category_thresholds[product_category] = {'count': 0, 'threshold': 75.0}
                category_thresholds[product_category]['count'] += 1
            
            brand = getBrand(product.store_product_name) if product.store_product_name else None
            if brand:
                if brand not in brand_thresholds:
                    brand_thresholds[brand] = {'count': 0, 'threshold': 75.0}
                brand_thresholds[brand]['count'] += 1
        
        # Adjust threshold based on brand frequency
        for brand, data in brand_thresholds.items():
            if data['count'] > 20:
                brand_thresholds[brand]['threshold'] = 80.0
            elif data['count'] > 10:
                brand_thresholds[brand]['threshold'] = 77.0
        
        # Adjust threshold based on category frequency
        for category_id, data in category_thresholds.items():
            if data['count'] > 30:
                category_thresholds[category_id]['threshold'] = 78.0
        
        match_metrics = {
            "total_processed": 0,
            "high_confidence_matches": 0,
            "medium_confidence_matches": 0,
            "low_confidence_matches": 0,
            "failed_matches": 0,
            "confidence_scores": []
        }
        
        # Dictionary to track already matched products
        processed_products = {}
        
        # For periodic progress reporting
        total_to_process = len(unmatched_products)
        progress_interval = max(1, total_to_process // 10)  # Show progress approximately 10 times

        for idx, first_candidate in enumerate(unmatched_products):
            # Show progress periodically
            if idx % progress_interval == 0:
                logger.info(f"Processing progress: {idx}/{total_to_process} ({idx/total_to_process*100:.1f}%)")
                
            if not first_candidate.store_product_name or "SKIP" in first_candidate.store_product_name:
                continue
                
            # Create safe unique identifier for the product
            # Use only guaranteed existing attributes
            candidate_id = f"{first_candidate.store_id}_{hash(first_candidate.store_product_name)}"
                
            # Check if this product has already been processed
            if candidate_id in processed_products:
                continue

            normalized_name = normalize_string(first_candidate.store_product_name)
            
            similar_products = db_session.query(ProductStoreData).filter(
                ProductStoreData.store_id != first_candidate.store_id,
                func.similarity(
                    func.lower(ProductStoreData.store_product_name), 
                    normalized_name.lower()
                ) > 0.3
            ).order_by(
                func.similarity(
                    func.lower(ProductStoreData.store_product_name), 
                    normalized_name.lower()
                ).desc()
            ).limit(20).all()
            
            # Filter products that have already been processed
            filtered_similar_products = []
            for p in similar_products:
                # Create identifier for each similar product in the same way
                p_id = f"{p.store_id}_{hash(p.store_product_name)}"
                
                if p_id not in processed_products:
                    filtered_similar_products.append(p)
            
            similar_products = filtered_similar_products

            first_main_words, first_attributes = estonian_nlp.extract_features(first_candidate.store_product_name)
            
            best_match = None
            best_score = 0.0
            best_metrics = {}

            for second_candidate in similar_products:
                second_main_words, second_attributes = estonian_nlp.extract_features(second_candidate.store_product_name)
                
                # Compare products with updated method
                nlp_similarity = estonian_nlp.compare_products(
                    first_candidate.store_product_name, 
                    second_candidate.store_product_name
                )
                
                # For metadata, get brand separately, only when needed
                if nlp_similarity > best_score:
                    best_match = second_candidate
                    best_score = nlp_similarity
                    
                    # Additional analysis to fill metrics
                    word_similarity = estonian_nlp._improved_word_similarity(first_main_words, second_main_words) * 100
                    attribute_score = estonian_nlp._attribute_similarity(first_attributes, second_attributes) * 100
                    unique_words_score = estonian_nlp._simple_unique_words(
                        first_candidate.store_product_name, 
                        second_candidate.store_product_name
                    ) * 100
                    
                    best_metrics = {
                        'word_similarity': word_similarity,
                        'attribute_score': attribute_score,
                        'unique_words_score': unique_words_score,
                        'first_main_words': first_main_words,
                        'second_main_words': second_main_words,
                        'first_attributes': first_attributes,
                        'second_attributes': second_attributes
                    }
                
                # Save to Excel with high threshold to reduce writing
                if nlp_similarity > 70.0:
                    save_potential_matches_to_file(
                        first_candidate.store_product_name,
                        second_candidate.store_product_name,
                        nlp_similarity,
                        best_metrics
                    )
            
            match_threshold = 82.0
            
            product_category = getattr(first_candidate, "category_id", None)
            if product_category and product_category in category_thresholds:
                match_threshold = category_thresholds[product_category]['threshold']
            
            if best_match and best_score >= 80.0:
                # Get brand only when needed for metadata
                brand = getBrand(first_candidate.store_product_name)
                
                nlp_features = {
                    "brand": brand,
                    "main_words": first_main_words,
                    "attributes": first_attributes
                }
                
                if not first_candidate.additional_attributes:
                    first_candidate.additional_attributes = {}
                first_candidate.additional_attributes["nlp_features"] = nlp_features

                # Mark both products as processed
                processed_products[candidate_id] = True
                
                # Create identifier for best_match the same way
                best_match_id = f"{best_match.store_id}_{hash(best_match.store_product_name)}"
                processed_products[best_match_id] = True

                if best_match.product_id is not None:
                    first_candidate.product_id = best_match.product_id
                    matched_count += 1
                elif first_candidate.product_id is not None:
                    best_match.product_id = first_candidate.product_id
                    matched_count += 1
                else:
                    new_product = create_new_product(first_candidate)
                    best_match.product_id = new_product.product_id
                    created_count += 1

                first_candidate.last_matched = func.now()
                best_match.last_matched = func.now()
                db_session.commit()
            else:
                # Mark product as processed
                processed_products[candidate_id] = True
                
                new_product = create_new_product(first_candidate)
                created_count += 1
                db_session.commit()

            match_metrics["total_processed"] += 1
            if best_score >= 80.0:
                match_metrics["high_confidence_matches"] += 1
                match_metrics["confidence_scores"].append(best_score)
            elif best_score >= 75.0:
                match_metrics["medium_confidence_matches"] += 1
                match_metrics["confidence_scores"].append(best_score) 
            elif best_score >= 50.0:
                match_metrics["low_confidence_matches"] += 1
                match_metrics["confidence_scores"].append(best_score)
            else:
                match_metrics["failed_matches"] += 1

        logger.info(f"""NLP matching completed:
            - Matched: {matched_count} products
            - New products created: {created_count} products
            - Success rate: {((matched_count + created_count) / len(unmatched_products) * 100):.1f}%""")

        return {
            "matched": matched_count,
            "created": created_count
        }

    except Exception as e:
        logger.error(f"Error during NLP matching process: {str(e)}", exc_info=True)
        db_session.rollback()
        raise

