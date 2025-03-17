from datetime import timedelta, datetime
import re
from sqlalchemy import func, or_
from app.core.logger import setup_logger
from app.models.product_store_data import ProductStoreData
from app.models.unit import Unit
from app import schemas
from app.services import product_service
from app.utils.brands import getBrand
import nltk
from collections import Counter
import string
import os
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
import io

logger = setup_logger("app.ai.matcher")

class EstonianProductNLP:
    def __init__(self):
        self.common_words = set()
        self.taste_markers = []
        self.units_cache = {}
        self.dynamic_common_words = {}
        self.common_words_threshold = 0.2
        self.min_word_length = 2
        self.dynamic_taste_markers = {}
        self.dynamic_taste_words = {}
        
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
            
        taste_patterns = []
        
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
        
        logger.info(f"Updated taste markers list. Total markers: {len(self.taste_markers)}")
        
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
        
        logger.info(f"Updated common words list. Total words: {len(self.common_words)}")
        
    def extract_features(self, text):
        if not text:
            return None, [], None, [], []
            
        text = text.lower()
        brand = getBrand(text)
        
        main_part = text
        
        bracket_pattern = re.compile(r'\(([^)]*)\)')
        bracket_matches = bracket_pattern.findall(text)
        attributes = []
        for match in bracket_matches:
            attributes.extend(self._tokenize(match))
            main_part = bracket_pattern.sub('', main_part)
        
        comma_parts = main_part.split(',')
        if len(comma_parts) > 1:
            main_part = comma_parts[0]
            for part in comma_parts[1:]:
                attributes.extend(self._tokenize(part))
        
        main_tokens = self._tokenize(main_part)
        
        product_type_indicators = self._extract_type_indicators(text, main_tokens)
        
        main_words = []
        for token in main_tokens:
            if token not in self.common_words and (not brand or token not in brand.lower()):
                if self._is_attribute(token) or self._could_be_important_attribute(token, text):
                    attributes.append(token)
                else:
                    main_words.append(token)
        
        taste = self._extract_taste(text, main_tokens)
        
        return brand, main_words, taste, attributes, product_type_indicators
    
    def _extract_type_indicators(self, text, tokens):
        """Extracts key words that characterize product type"""
        type_indicators = []
        
        for token in tokens:
            if len(token) > 6:
                if token.lower() in self.dynamic_common_words:
                    frequency = self.dynamic_common_words[token.lower()]
                    if frequency < 0.1:
                        type_indicators.append(token)
                else:
                    type_indicators.append(token)
        
        if not type_indicators and tokens:
            type_indicators = tokens[:2]
        
        return type_indicators
    
    def _is_attribute(self, token):
        """Dynamically determines if a token is an attribute"""
        has_digits = any(char.isdigit() for char in token)
        
        has_special = any(char in token for char in '/-+%')
        
        has_units = False
        unit_suffixes = ['g', 'kg', 'l', 'ml', 'mm', 'cm']
        for unit in unit_suffixes:
            if unit in token.lower():
                has_units = True
                break
        
        weight_pattern = re.compile(r'\d+[.,]?\d*\s*(?:g|kg|l|ml|мл|л|гр|кг)')
        is_weight = bool(weight_pattern.search(token.lower()))
        
        percent_pattern = re.compile(r'\d+[.,]?\d*\s*%')
        is_percent = bool(percent_pattern.search(token.lower()))
        
        return has_digits or has_special or has_units or is_weight or is_percent
    
    def _extract_taste(self, text, tokens):
        for marker in self.taste_markers:
            if marker in text:
                idx = text.find(marker)
                if idx >= 0:
                    after_taste = text[idx + len(marker):].strip()
                    taste_words = [w for w in after_taste.split()[:2] if w not in self.common_words]
                    if taste_words:
                        return ' '.join(taste_words)
                    
        for taste, count in sorted(self.dynamic_taste_words.items(), key=lambda x: x[1], reverse=True):
            if count > 5 and taste in text:
                return taste
                
        return None
    
    def _tokenize(self, text):
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        tokens = [word.strip() for word in text.split() if word.strip()]
        return tokens
    
    def compare_products(self, product1, product2):
        """Compares two products based on extracted features"""
        normalized_product1 = product1.lower().strip() if product1 else None
        normalized_product2 = product2.lower().strip() if product2 else None
        
        brand1, main_words1, taste1, attributes1, type_indicators1 = self.extract_features(normalized_product1)
        brand2, main_words2, taste2, attributes2, type_indicators2 = self.extract_features(normalized_product2)
        
        if brand1 != brand2:
            return 0.0
            
        unique_words_score = self._compare_unique_words(normalized_product1, normalized_product2)
        if unique_words_score < 0.3:
            return 0.0
        
        word_similarity = self._calculate_word_similarity(main_words1, main_words2)
        
        taste_score = 0.0
        if taste1 and taste2:
            if taste1 == taste2:
                taste_score = 1.0
            else:
                taste_similarity = self._levenshtein_similarity(taste1, taste2)
                taste_score = taste_similarity if taste_similarity > 0.7 else 0.0
        elif not taste1 and not taste2:
            taste_score = 0.7
        else:
            taste_score = 0.0
        
        attribute_score = 0.0
        if attributes1 and attributes2:
            attribute_score = self._calculate_attribute_similarity(attributes1, attributes2)
        elif not attributes1 and not attributes2:
            attribute_score = 1.0
        else:
            attribute_score = 0.1
        
        return (
            word_similarity * 0.2 +
            attribute_score * 0.3 +
            taste_score * 0.2 +
            unique_words_score * 0.3
        )
    
    def _compare_unique_words(self, product1, product2):
        """Compares rare and unique words in product names"""
        tokens1 = self._tokenize(product1)
        tokens2 = self._tokenize(product2)
        
        rare_words1 = []
        rare_words2 = []
        
        for token in tokens1:
            if len(token) > 5:
                if token in self.dynamic_common_words and self.dynamic_common_words[token] < 0.1:
                    rare_words1.append(token)
                elif token not in self.dynamic_common_words:
                    rare_words1.append(token)
        
        for token in tokens2:
            if len(token) > 5:
                if token in self.dynamic_common_words and self.dynamic_common_words[token] < 0.1:
                    rare_words2.append(token)
                elif token not in self.dynamic_common_words:
                    rare_words2.append(token)
        
        if not rare_words1 or not rare_words2:
            return 0.5
        
        best_matches = 0
        total_pairs = max(len(rare_words1), len(rare_words2))
        
        for word1 in rare_words1:
            max_similarity = 0
            for word2 in rare_words2:
                similarity = self._levenshtein_similarity(word1, word2)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity > 0.7:
                best_matches += 1
        
        return best_matches / total_pairs if total_pairs > 0 else 0.5
    
    def _calculate_attribute_similarity(self, attributes1, attributes2):
        """Improved attribute comparison considering their importance"""
        if not attributes1 or not attributes2:
            return 0.0
        
        weighted_matches = 0
        total_weight = 0
        
        for attr1 in attributes1:
            weight = min(len(attr1) / 5, 2.0)
            if attr1 in self.dynamic_common_words:
                freq = self.dynamic_common_words[attr1]
                if freq < 0.05:
                    weight *= 2
                elif freq < 0.1:
                    weight *= 1.5
            else:
                weight *= 2
            
            max_similarity = 0
            for attr2 in attributes2:
                similarity = self._levenshtein_similarity(attr1, attr2)
                max_similarity = max(max_similarity, similarity)
            
            weighted_matches += max_similarity * weight
            total_weight += weight
        
        return weighted_matches / total_weight if total_weight > 0 else 0.0
    
    def _calculate_word_similarity(self, words1, words2):
        if not words1 or not words2:
            return 0.0
            
        total_similarity = 0
        total_pairs = 0
        
        length_weights = {}
        for word in words1 + words2:
            if len(word) > 10:
                length_weights[word] = 0.6
            elif len(word) > 7:
                length_weights[word] = 0.8
            else:
                length_weights[word] = 1.0
        
        for word1 in words1:
            max_similarity = 0
            max_word = None
            for word2 in words2:
                similarity = self._levenshtein_similarity(word1, word2)
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_word = word2
            
            if max_similarity > 0.7:
                word_weight = min(length_weights[word1], length_weights.get(max_word, 1.0))
                total_similarity += max_similarity * word_weight
                total_pairs += word_weight
            
        for word2 in words2:
            max_similarity = 0
            max_word = None
            for word1 in words1:
                similarity = self._levenshtein_similarity(word2, word1)
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_word = word1
            
            if max_similarity > 0.7:
                # Применяем вес в зависимости от длины слова
                word_weight = min(length_weights[word2], length_weights.get(max_word, 1.0))
                total_similarity += max_similarity * word_weight
                total_pairs += word_weight
            
        if total_pairs > 0:
            return total_similarity / total_pairs
        
        set1 = set(words1)
        set2 = set(words2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _levenshtein_similarity(self, str1, str2):
        """Calculates normalized similarity based on Levenshtein distance"""
        if not str1 or not str2:
            return 0.0
            
        distance = self._levenshtein_distance(str1, str2)
        
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
            
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, str1, str2):
        """Calculates Levenshtein distance between two strings"""
        m, n = len(str1), len(str2)
        
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
            
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if str1[i-1] == str2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
                
        return dp[m][n]
    
    def extract_package_multiplier(self, text):
        """Extracts package multiplier (e.g., 6x from '6X330ml')"""
        multiplier_pattern = re.compile(r'(\d+)\s*[xX*×]\s*\d+')
        match = multiplier_pattern.search(text)
        if match:
            return int(match.group(1))
        
        pack_pattern = re.compile(r'(\d+)\s*(?:pack|pakk|tk)')
        match = pack_pattern.search(text)
        if match:
            return int(match.group(1))
        
        return 1

    def convert_to_base_unit(self, weight_value, unit_id, db_session, product_name=None):
        """Converts product weight to base unit considering package multiplier"""
        if not weight_value or not unit_id:
            return None
            
        if unit_id not in self.units_cache:
            unit = db_session.query(Unit).filter(Unit.unit_id == unit_id).first()
            if not unit:
                return None
            self.units_cache[unit_id] = unit.conversion_factor
            
        conversion_factor = self.units_cache[unit_id]
        
        if hasattr(conversion_factor, 'as_tuple'):
            conversion_factor = float(conversion_factor)
        
        multiplier = 1
        if product_name:
            multiplier = self.extract_package_multiplier(product_name)
        
        return float(weight_value) * conversion_factor * multiplier
    
    def are_weights_compatible(self, weight1, unit_id1, weight2, unit_id2, db_session, tolerance=0.05, product_name1=None, product_name2=None):
        """Checks compatibility of product weights considering allowable tolerance and package multiplier"""
        if not weight1 or not weight2 or not unit_id1 or not unit_id2:
            return False
            
        base_weight1 = self.convert_to_base_unit(weight1, unit_id1, db_session, product_name1)
        base_weight2 = self.convert_to_base_unit(weight2, unit_id2, db_session, product_name2)
        
        if not base_weight1 or not base_weight2:
            return False
            
        diff_percent = abs(base_weight1 - base_weight2) / max(base_weight1, base_weight2)
        
        logger.debug(f"Weight comparison: {base_weight1} vs {base_weight2}, diff: {diff_percent*100:.1f}%")
        
        return diff_percent <= tolerance

    def _check_important_attributes(self, attributes1, attributes2):
        """Checks for important attribute differences between products"""
        if (attributes1 and not attributes2) or (not attributes1 and attributes2):
            return True
        
        if not attributes1 and not attributes2:
            return False
        
        if abs(len(attributes1) - len(attributes2)) > 0:
            return True
        
        for attr1 in attributes1:
            found_match = False
            for attr2 in attributes2:
                if self._levenshtein_similarity(attr1, attr2) > 0.9:
                    found_match = True
                    break
            if not found_match:
                return True
        
        return False

    def _calculate_type_match(self, type_indicators1, type_indicators2):
        """Calculates product type match degree without hardcoded lists"""
        if not type_indicators1 or not type_indicators2:
            return 0.5
        
        if len(type_indicators1) > 0 and len(type_indicators2) > 0:
            first_word_match = self._levenshtein_similarity(type_indicators1[0], type_indicators2[0])
            
            if first_word_match < 0.5:
                return 0.1
        
        total_score = 0.0
        matches = 0
        
        for word1 in type_indicators1:
            best_match = 0.0
            for word2 in type_indicators2:
                similarity = self._levenshtein_similarity(word1, word2)
                best_match = max(best_match, similarity)
            
            if best_match > 0:
                total_score += best_match
                matches += 1
        
        if matches > 0:
            return total_score / matches
        
        return 0.5

    def _could_be_important_attribute(self, token, full_text):
        """Checks if a token could be an important product attribute"""
        position = full_text.lower().find(token.lower())
        
        words = full_text.lower().split()
        if words and len(words) >= 1 and token.lower() == words[0].lower():
            return True
        
        if position >= 0 and len(token) >= 3:
            if not any(char.isdigit() for char in token):
                if position < len(full_text) / 3:
                    return True
                
                if position > 0 and position + len(token) < len(full_text):
                    before = full_text[position-1]
                    after = full_text[position+len(token)]
                    if (before.isspace() or before in ',.(/-') and (after.isspace() or after in ',.)/-'):
                        return True
        
        if token.lower() in self.dynamic_common_words:
            frequency = self.dynamic_common_words[token.lower()]
            if frequency < 0.05:
                return True
        
        estonian_adj_endings = ['ne', 'line', 'lik', 'lane', 'ke', 'mine', 'v', 'tu', 'kas']
        for ending in estonian_adj_endings:
            if len(token) > len(ending) + 1 and token.endswith(ending):
                return True
        
        return False

    def _has_digits(self, text):
        """Checks for digits in text"""
        return any(char.isdigit() for char in text)

estonian_nlp = EstonianProductNLP()

# Function to save potential matches to Excel file
def save_potential_matches_to_file(first_product, second_product, score):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"potential_matches_{timestamp}.xlsx"
        
        # Create a new workbook if file doesn't exist
        if not os.path.isfile(filename):
            wb = Workbook()
            ws = wb.active
            ws.title = "Potential Matches"
            headers = ['First Product', 'Second Product', 'Score', 'Timestamp']
            for col_num, header in enumerate(headers, 1):
                col_letter = get_column_letter(col_num)
                ws[f'{col_letter}1'] = header
                ws.column_dimensions[col_letter].width = 40  # Set column width
        else:
            # Load existing workbook
            wb = load_workbook(filename)
            ws = wb["Potential Matches"]
        
        # Add new row
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [first_product, second_product, f"{score:.1f}%", current_time]
        ws.append(row)
        
        # Auto-adjust columns width for better readability
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                if cell.value:
                    cell_length = len(str(cell.value))
                    if cell_length > max_length:
                        max_length = cell_length
            adjusted_width = max_length + 2
            ws.column_dimensions[column_letter].width = adjusted_width if adjusted_width > 15 else 15
        
        # Save the file
        wb.save(filename)
            
        return True
    except Exception as e:
        logger.error(f"Error saving potential match to file: {str(e)}")
        return False

def run_matching(db_session):
    """Run matching based on weight and title similarity with Estonian NLP"""
    try:
        logger.info("Starting product matching with Estonian NLP")

        estonian_nlp.initialize_from_data(db_session)

        def normalize_string(text):
            if text:
                return text.lower().strip()
            return text
            
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
                    first_candidate.product_id = ean_match.product_id
                    first_candidate.last_matched = func.now()
                    ean_matched_count += 1
                    db_session.commit()
                    logger.info(f"EAN match found: {first_candidate.store_product_name} → {ean_match.store_product_name} (EAN: {first_candidate.ean})")
                    
                    product = db_session.query(schemas.Product).filter(
                        schemas.Product.product_id == ean_match.product_id
                    ).first()
                    if product and not product.barcode:
                        product.barcode = first_candidate.ean
                        db_session.commit()
                        logger.info(f"Updated product barcode: {product.product_id} → {first_candidate.ean}")
                
                elif first_candidate.product_id is not None:
                    for match in ean_matches:
                        match.product_id = first_candidate.product_id
                        match.last_matched = func.now()
                    ean_matched_count += len(ean_matches)
                    db_session.commit()
                    logger.info(f"Applied existing product_id to EAN matches: {first_candidate.ean}")
                else:
                    new_product = product_service.create(db_session, schemas.ProductCreate(
                        name=first_candidate.store_product_name,
                        weight_value=first_candidate.store_weight_value,
                        unit_id=first_candidate.store_unit_id,
                        barcode=first_candidate.ean
                    ))
                    first_candidate.product_id = new_product.product_id
                    first_candidate.last_matched = func.now()
                    
                    for match in ean_matches:
                        match.product_id = new_product.product_id
                        match.last_matched = func.now()
                    
                    ean_matched_count += 1 + len(ean_matches)
                    db_session.commit()
                    logger.info(f"Created new product from EAN: {first_candidate.ean}")
        
        logger.info(f"EAN matching completed: {ean_matched_count} matches found")
        
        unmatched_products = [p for p in unmatched_products if p.product_id is None]
        
        category_index = {}
        for product in unmatched_products:
            category_key = getattr(product, "category_id", None) 
            if category_key is None:
                category_key = "default"
                
            if category_key not in category_index:
                category_index[category_key] = []
            category_index[category_key].append(product)
            
        logger.info(f"Found {len(unmatched_products)} unmatched products in {len(category_index)} categories")

        if not unmatched_products:
            logger.info("No unmatched products found")
            return

        logger.info(f"Found {len(unmatched_products)} unmatched products for NLP analysis")

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

        for first_candidate in unmatched_products:
            if not first_candidate.store_product_name or "SKIP" in first_candidate.store_product_name:
                continue

            normalized_name = normalize_string(first_candidate.store_product_name)
            
            similar_products = db_session.query(ProductStoreData).filter(
                ProductStoreData.store_id != first_candidate.store_id,
                func.similarity(func.lower(ProductStoreData.store_product_name), normalized_name) > 0.3
            ).order_by(
                func.similarity(func.lower(ProductStoreData.store_product_name), normalized_name).desc()
            ).limit(20).all()

            first_brand, first_main_words, first_taste, first_attributes, first_type_indicators = estonian_nlp.extract_features(first_candidate.store_product_name)

            best_match = None
            best_score = 0.0

            for second_candidate in similar_products:
                second_brand, second_main_words, second_taste, second_attributes, second_type_indicators = estonian_nlp.extract_features(second_candidate.store_product_name)

                if first_brand != second_brand:
                    continue

                nlp_similarity = estonian_nlp.compare_products(
                    first_candidate.store_product_name, 
                    second_candidate.store_product_name
                )
                
                weights_compatible = estonian_nlp.are_weights_compatible(
                    first_candidate.store_weight_value,
                    first_candidate.store_unit_id,
                    second_candidate.store_weight_value,
                    second_candidate.store_unit_id,
                    db_session,
                    product_name1=first_candidate.store_product_name,
                    product_name2=second_candidate.store_product_name
                )
                
                if not weights_compatible and first_candidate.store_weight_value and second_candidate.store_weight_value:
                    continue
                
                weight_similarity = 0.0
                if first_candidate.store_weight_value and second_candidate.store_weight_value:
                    base_weight1 = estonian_nlp.convert_to_base_unit(
                        first_candidate.store_weight_value, 
                        first_candidate.store_unit_id, 
                        db_session,
                        first_candidate.store_product_name
                    )
                    base_weight2 = estonian_nlp.convert_to_base_unit(
                        second_candidate.store_weight_value, 
                        second_candidate.store_unit_id, 
                        db_session,
                        second_candidate.store_product_name
                    )
                    
                    if base_weight1 and base_weight2:
                        weight_diff = abs(base_weight2 - base_weight1)
                        weight_similarity = 1 - (weight_diff / max(base_weight1, 0.0001))

                total_score = (nlp_similarity * 0.7 + weight_similarity * 0.3) * 100

                if total_score > best_score:
                    best_match = second_candidate
                    best_score = total_score
            
            match_threshold = 82.0
            
            product_category = getattr(first_candidate, "category_id", None)
            if product_category and product_category in category_thresholds:
                match_threshold = category_thresholds[product_category]['threshold']
            
            if first_brand and first_brand in brand_thresholds:
                match_threshold = brand_thresholds[first_brand]['threshold']

            # Save matches between 65% and 80% to file (no console logging)
            if best_match and best_score < 80.0 and best_score >= 65.0:
                save_potential_matches_to_file(
                    first_candidate.store_product_name,
                    best_match.store_product_name,
                    best_score
                )

            if best_match and best_score >= 80.0:
                logger.info(f"NLP match found for {first_candidate.store_product_name} and {best_match.store_product_name} with score {best_score:.1f}%")

                nlp_features = {
                    "brand": first_brand,
                    "main_words": first_main_words,
                    "taste": first_taste,
                    "attributes": first_attributes
                }
                
                if not first_candidate.additional_attributes:
                    first_candidate.additional_attributes = {}
                first_candidate.additional_attributes["nlp_features"] = nlp_features

                if best_match.product_id is not None:
                    first_candidate.product_id = best_match.product_id
                    matched_count += 1
                elif first_candidate.product_id is not None:
                    best_match.product_id = first_candidate.product_id
                    matched_count += 1
                else:
                    new_product = product_service.create(db_session, schemas.ProductCreate(
                        name=first_candidate.store_product_name,
                        weight_value=first_candidate.store_weight_value,
                        unit_id=first_candidate.store_unit_id
                    ))
                    first_candidate.product_id = new_product.product_id
                    best_match.product_id = new_product.product_id
                    created_count += 1

                first_candidate.last_matched = func.now()
                best_match.last_matched = func.now()
                db_session.commit()
            else:
                new_product = product_service.create(db_session, schemas.ProductCreate(
                    name=first_candidate.store_product_name,
                    weight_value=first_candidate.store_weight_value,
                    unit_id=first_candidate.store_unit_id
                ))
                
                nlp_features = {
                    "brand": first_brand,
                    "main_words": first_main_words,
                    "taste": first_taste,
                    "attributes": first_attributes
                }
                
                if not first_candidate.additional_attributes:
                    first_candidate.additional_attributes = {}
                first_candidate.additional_attributes["nlp_features"] = nlp_features
                
                first_candidate.product_id = new_product.product_id
                first_candidate.last_matched = func.now()
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

        logger.info(f"""NLP Matching completed:
            - Matched: {matched_count} products
            - Created new: {created_count} products
            - Success rate: {((matched_count + created_count) / len(unmatched_products) * 100):.1f}%""")

        logger.info(f"Match metrics: {match_metrics}")

        return {
            "matched": matched_count,
            "created": created_count
        }

    except Exception as e:
        logger.error(f"Error during NLP matching process: {str(e)}", exc_info=True)
        db_session.rollback()
        raise

