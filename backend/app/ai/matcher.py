from datetime import timedelta, datetime
import re
from sqlalchemy import func, or_
from app.core.logger import setup_logger
from app.models.product_store_data import ProductStoreData
from app.models.unit import Unit
from app import schemas
from app.services import product_service
from app.utils.brands import getBrand, normalize_string as normalize_brand
import nltk
from collections import Counter
import string
import os
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
import io
import csv
import pandas as pd

logger = setup_logger("app.ai.matcher")

def normalize_string(text):
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
        """Инициализируем класс с дополнительными атрибутами"""
        # Базовая инициализация
        self.common_words = set()
        self.taste_markers = []
        self.units_cache = {}
        self.dynamic_common_words = {}
        self.common_words_threshold = 0.2
        self.min_word_length = 2
        self.dynamic_taste_markers = {}
        self.dynamic_taste_words = {}
        
        # Регулярное выражение для весов
        self.weight_pattern = r'\b\d+x\d+[a-z]*\b|\b\d+[a-z]*\b'
        
        # Динамические словари и другие атрибуты
        self.brand_candidates = set()
        self.dynamic_brands = {}
        self.probable_tastes = []
        self.taste_indicators = []
        self.probable_units = []
        self.product_samples = []
        self.product_type_indicators = set()
        
        # Коэффициенты для сравнения
        self.comparison_weights = {
            'word': 0.6,
            'attribute': 0.3,
            'unique': 0.1
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
        """Извлекает характеристики из текста продукта"""
        if not text:
            return None, [], None, [], []
            
        normalized_text = normalize_string(text)
        
        # Получение бренда (оставляем для совместимости, но не используем)
        brand = None
        
        # Разделяем текст на токены
        tokens = self._tokenize(normalized_text)
        
        # Извлекаем атрибуты (слова в скобках, после запятых и т.д.)
        attributes = []
        
        # Ищем слова в скобках
        bracket_pattern = re.compile(r'\((.*?)\)')
        bracket_matches = bracket_pattern.findall(normalized_text)
        for match in bracket_matches:
            # Добавляем строковые значения, а не кортежи
            attributes.extend([token for token in self._tokenize(match)])
            normalized_text = bracket_pattern.sub('', normalized_text)
        
        # Ищем части после запятой
        comma_parts = normalized_text.split(',')
        if len(comma_parts) > 1:
            normalized_text = comma_parts[0]
            for part in comma_parts[1:]:
                # Убедимся, что добавляем строки
                attributes.extend([token for token in self._tokenize(part)])
        
        # Определяем шаблон для весов напрямую, не используя self.weight_pattern
        weight_pattern = r'\b\d+x\d+[a-z]*\b|\b\d+[a-z]*\b'
        
        # Ищем веса и единицы измерения
        weight_matches = re.findall(weight_pattern, normalized_text)
        for match in weight_matches:
            if match and match not in attributes:
                attributes.append(match)
        
        # Заглушка для вкуса (оставляем для совместимости, но не используем)
        taste = None
        
        # Получаем основные слова
        main_words = self._tokenize(normalized_text)
        
        # Определяем индикаторы типа продукта
        type_indicators = []
        # Проверяем, что атрибут существует и не пустой
        if hasattr(self, 'product_type_indicators') and self.product_type_indicators:
            for word in main_words:
                if word in self.product_type_indicators:
                    type_indicators.append(word)
        
        # Удаляем дубликаты и убеждаемся, что все элементы являются строками
        main_words = [str(word) for word in list(set(main_words))]
        attributes = [str(attr) for attr in list(set(attributes))]
        
        return brand, main_words, taste, attributes, type_indicators
    
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
        """Tokenize and sort words"""
        text = normalize_string(text)
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        tokens = [word.strip() for word in text.split() if word.strip()]
        tokens.sort()  # Sort tokens
        return tokens
    
    def _improved_word_similarity(self, words1, words2):
        """
        Улучшенное сравнение наборов слов с учетом частичных совпадений,
        нормализации слов и обязательного наличия всех критических слов.
        """
        if not words1 or not words2:
            return 0.0

        # Нормализуем слова и обрабатываем сокращения
        norm_words1 = []
        for w in words1:
            # Удаляем точку в конце, если она есть (для сокращений)
            w = w.rstrip('.')
            norm_words1.append(normalize_string(w))
        
        norm_words2 = []
        for w in words2:
            # Удаляем точку в конце, если она есть (для сокращений)
            w = w.rstrip('.')
            norm_words2.append(normalize_string(w))
        
        # Находим критические слова (слова длиннее 3 символов, которые могут указывать на 
        # характеристики продукта - вкусы, модели и т.д.)
        critical_words1 = [w for w in norm_words1 if len(w) > 3 and not self._is_common_word(w)]
        critical_words2 = [w for w in norm_words2 if len(w) > 3 and not self._is_common_word(w)]
        
        # Проверяем каждое критическое слово из первого набора
        for word1 in critical_words1:
            best_match_score = 0
            for word2 in critical_words2:
                # Проверка на сокращения
                if word1.startswith(word2) and len(word2) >= 3:
                    # Если слово2 является началом слова1 и достаточно длинное
                    similarity = 0.9
                elif word2.startswith(word1) and len(word1) >= 3:
                    # Если слово1 является началом слова2 и достаточно длинное
                    similarity = 0.9
                # Рассчитаем сходство слов
                elif word1 == word2:
                    similarity = 1.0
                elif word1 in word2 or word2 in word1:
                    # Одно слово содержится в другом полностью
                    similarity = 0.9
                else:
                    similarity = self._levenshtein_similarity(word1, word2)
                
                best_match_score = max(best_match_score, similarity)
            
            # Если для важного слова нет достаточно близкого соответствия (< 80%), 
            # считаем продукты разными
            if best_match_score < 0.8:
                return 0.0
        
        # То же самое для второго набора слов
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
        
        # Если все критические слова имеют соответствия, рассчитываем общую схожесть
        total_similarity = 0.0
        max_possible = max(len(norm_words1), len(norm_words2))
        
        if max_possible == 0:
            return 0.0
        
        # Считаем общее количество совпадающих слов
        matches = 0
        for word1 in norm_words1:
            for word2 in norm_words2:
                if word1 == word2 or word1 in word2 or word2 in word1 or self._levenshtein_similarity(word1, word2) > 0.8:
                    matches += 1
                    break
        
        return matches / max_possible

    def _is_common_word(self, word):
        """Проверяет, является ли слово общим словом"""
        # Нормализуем слово перед проверкой
        word = normalize_string(word.lower())
        
        return (word in self.common_words or 
                word in self.dynamic_common_words and 
                self.dynamic_common_words[word] >= self.common_words_threshold)

    def _extract_taste_and_model(self, text):
        """Извлекает информацию о вкусе или модели продукта"""
        if not text:
            return None
        
        # Искать информацию о вкусе
        for marker in self.taste_markers:
            if marker in text.lower():
                idx = text.lower().find(marker)
                if idx >= 0:
                    after_marker = text[idx + len(marker):].strip()
                    before_marker = text[:idx].strip()
                    
                    # Проверяем слова после маркера
                    taste_words = [w for w in after_marker.split()[:2] 
                                  if w and len(w) > 2 and not self._is_common_word(w)]
                    if taste_words:
                        return ' '.join(taste_words)
                    
                    # Проверяем слова перед маркером
                    taste_words = [w for w in before_marker.split()[-2:] 
                                  if w and len(w) > 2 and not self._is_common_word(w)]
                    if taste_words:
                        return ' '.join(taste_words)
        
        # Искать модели и коды продуктов
        model_patterns = [
            r'[A-Z0-9]{2,}-[A-Z0-9]{2,}', # Формат XX-XX
            r'[A-Z]{2,}\d{2,}',           # Формат XXnn
            r'\d{3,}[A-Z]+',              # Формат nnnX
            r'[A-Z]{1,2}\d{1,2}'          # Формат Xn
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        return None

    def compare_products(self, product1, product2):
        """Сравнивает два продукта с улучшенной логикой обработки критических различий"""
        brand1, main_words1, taste1, attributes1, type_indicators1 = self.extract_features(product1)
        brand2, main_words2, taste2, attributes2, type_indicators2 = self.extract_features(product2)
        
        # Извлекаем вкусы и модели, которые критичны для сравнения
        taste_model1 = self._extract_taste_and_model(product1)
        taste_model2 = self._extract_taste_and_model(product2)
        
        # Если у обоих продуктов есть вкус/модель, но они разные - продукты разные
        if taste_model1 and taste_model2 and taste_model1 != taste_model2:
            # Проверяем частичное совпадение для учета разных форматов записи
            if self._levenshtein_similarity(taste_model1, taste_model2) < 0.7:
                return 0.0
        
        # Проверка веса/объема продукта, если в названии есть цифры с единицами измерения
        weight_pattern = r'(\d+[.,]?\d*)\s*(г|g|кг|kg|л|l|мл|ml)'
        weight1 = re.findall(weight_pattern, product1.lower().replace(',', '.'))
        weight2 = re.findall(weight_pattern, product2.lower().replace(',', '.'))
        
        if weight1 and weight2 and weight1[0][0] != weight2[0][0]:
            return 0.0
        
        # Сравниваем основные слова с учетом критических атрибутов
        word_similarity = self._improved_word_similarity(main_words1, main_words2)
        
        # Если основные слова не совпадают достаточно хорошо, прекращаем сравнение
        if word_similarity < 0.3:
            return 0.0
        
        # Сравниваем атрибуты
        attribute_score = self._attribute_similarity(attributes1, attributes2)
        
        # Сравниваем уникальные слова
        unique_words_score = self._simple_unique_words(product1, product2)
        
        # Обновленное распределение весов
        total_score = (
            word_similarity * 0.4 + 
            attribute_score * 0.4 + 
            unique_words_score * 0.2
        )
        
        # Возвращаем процентное значение для лучшей читаемости
        return total_score * 100

    def _calculate_attribute_similarity(self, attributes1, attributes2):
        """Improved attribute comparison considering their importance"""
        if not attributes1 or not attributes2:
            return 0.0
        
        # Sort attributes
        attributes1 = sorted(attributes1)
        attributes2 = sorted(attributes2)
        
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
            
        # Sort word lists
        words1 = sorted(words1)
        words2 = sorted(words2)
        
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

    def _attribute_similarity(self, attributes1, attributes2):
        """
        Вычисляет сходство двух наборов атрибутов.
        Учитывает как точные, так и частичные совпадения.
        """
        if not attributes1 or not attributes2:
            # Если один из наборов пустой, возвращаем 0 или небольшое значение
            # Если оба набора пустые, это можно считать хорошим совпадением
            if not attributes1 and not attributes2:
                return 1.0
            return 0.1  # Небольшое значение, чтобы не отбрасывать продукты полностью
        
        # Нормализуем атрибуты
        norm_attrs1 = []
        for attr in attributes1:
            try:
                # Преобразуем в строку, если это не строка
                if not isinstance(attr, str):
                    attr = str(attr)
                norm_attrs1.append(normalize_string(attr))
            except:
                continue
            
        norm_attrs2 = []
        for attr in attributes2:
            try:
                # Преобразуем в строку, если это не строка
                if not isinstance(attr, str):
                    attr = str(attr)
                norm_attrs2.append(normalize_string(attr))
            except:
                continue
        
        # Если после нормализации наборы пустые
        if not norm_attrs1 or not norm_attrs2:
            if not norm_attrs1 and not norm_attrs2:
                return 1.0
            return 0.1
        
        # Находим точные совпадения
        exact_matches = set(norm_attrs1) & set(norm_attrs2)
        exact_match_count = len(exact_matches)
        
        # Ищем частичные совпадения для атрибутов, которые не совпали точно
        partial_matches_score = 0.0
        remaining_attrs1 = [a for a in norm_attrs1 if a not in exact_matches]
        remaining_attrs2 = [a for a in norm_attrs2 if a not in exact_matches]
        
        for attr1 in remaining_attrs1:
            best_partial_match = 0.0
            for attr2 in remaining_attrs2:
                # Пропускаем слишком короткие атрибуты
                if len(attr1) < 2 or len(attr2) < 2:
                    continue
                    
                # Используем расстояние Левенштейна для определения похожести
                if hasattr(self, '_levenshtein_similarity'):
                    similarity = self._levenshtein_similarity(attr1, attr2)
                else:
                    # Простое сравнение, если метод _levenshtein_similarity не доступен
                    if attr1 == attr2:
                        similarity = 1.0
                    else:
                        similarity = 0.0
                    
                if similarity > 0.8:  # Порог для частичного совпадения
                    best_partial_match = max(best_partial_match, similarity * 0.8)
            
            partial_matches_score += best_partial_match
        
        # Вычисляем итоговую оценку сходства
        total_possible = max(len(norm_attrs1), len(norm_attrs2))
        if total_possible == 0:
            return 1.0  # Оба набора пустые после фильтрации
        
        # Комбинируем точные и частичные совпадения
        similarity_score = (exact_match_count + partial_matches_score) / total_possible
        
        return similarity_score

    def _levenshtein_similarity(self, str1, str2):
        """
        Вычисляет меру сходства на основе расстояния Левенштейна.
        Возвращает значение от 0 до 1, где 1 - полное совпадение.
        """
        if not str1 or not str2:
            return 0.0
        
        # Нормализуем строки перед сравнением
        str1 = normalize_string(str1.lower())
        str2 = normalize_string(str2.lower())
        
        # Если строки идентичны, сразу возвращаем 1
        if str1 == str2:
            return 1.0
        
        # Создаем матрицу размером (len(str1)+1) x (len(str2)+1)
        matrix = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
        
        # Инициализируем первую строку и первый столбец
        for i in range(len(str1) + 1):
            matrix[i][0] = i
        for j in range(len(str2) + 1):
            matrix[0][j] = j
        
        # Заполняем матрицу
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                cost = 0 if str1[i-1] == str2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # Удаление
                    matrix[i][j-1] + 1,      # Вставка
                    matrix[i-1][j-1] + cost  # Замена или совпадение
                )
        
        # Получаем расстояние Левенштейна
        distance = matrix[len(str1)][len(str2)]
        
        # Вычисляем сходство как 1 - (расстояние / длина_более_длинной_строки)
        max_length = max(len(str1), len(str2))
        similarity = 1.0 - (distance / max_length if max_length > 0 else 0)
        
        return similarity

    def _simple_unique_words(self, product1, product2):
        """
        Сравнивает уникальные слова в названиях продуктов.
        Игнорирует общие слова и сосредотачивается на специфичных словах.
        """
        if not product1 or not product2:
            return 0.0
        
        # Нормализуем и разбиваем на слова
        words1 = [w.rstrip('.') for w in normalize_string(product1.lower()).split()]
        words2 = [w.rstrip('.') for w in normalize_string(product2.lower()).split()]
        
        # Отфильтровываем общие слова и слишком короткие слова
        unique_words1 = [w for w in words1 if len(w) > 3 and not self._is_common_word(w)]
        unique_words2 = [w for w in words2 if len(w) > 3 and not self._is_common_word(w)]
        
        # Если нет уникальных слов, считаем продукты похожими
        if not unique_words1 and not unique_words2:
            return 1.0
        elif not unique_words1 or not unique_words2:
            return 0.0
        
        # Находим совпадающие уникальные слова
        matches = 0
        for word1 in unique_words1:
            for word2 in unique_words2:
                # Нормализуем перед сравнением
                norm_word1 = normalize_string(word1)
                norm_word2 = normalize_string(word2)
                
                # Проверяем или точное совпадение, или содержание одного слова в другом
                if norm_word1 == norm_word2 or (len(norm_word1) > 4 and norm_word1 in norm_word2) or (len(norm_word2) > 4 and norm_word2 in norm_word1):
                    matches += 1
                    break
        
        # Вычисляем показатель совпадения
        max_unique = max(len(unique_words1), len(unique_words2))
        if max_unique > 0:
            return matches / max_unique
        
        return 0.5  # Возвращаем среднее значение, если нет уникальных слов

estonian_nlp = EstonianProductNLP()

# Функция сохранения совпадений в файл Excel
def save_potential_matches_to_file(first_product, second_product, score, metrics):
    """Сохраняет информацию о потенциальных совпадениях в Excel файл"""
    try:
        filepath = "potential_matches.xlsx"
        file_exists = os.path.isfile(filepath)
        
        # Создаем данные для записи
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
        
        # Создаем DataFrame из данных
        df = pd.DataFrame(data)
        
        # Если файл существует, загружаем его и добавляем новые данные
        if file_exists:
            try:
                existing_df = pd.read_excel(filepath)
                # Объединяем существующие данные с новыми
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                logger.error(f"Ошибка при чтении существующего файла Excel: {str(e)}")
                # Если не удалось прочитать существующий файл, создаем новый
        
        # Сохраняем DataFrame в Excel файл
        df.to_excel(filepath, index=False, engine='openpyxl')
        
        logger.info(f"Данные о совпадении успешно сохранены в {filepath}")
        
    except Exception as e:
        logger.error(f"Ошибка сохранения совпадений в Excel файл: {str(e)}")

def run_matching(db_session):
    """Run matching based on weight and title similarity with Estonian NLP"""
    try:
        logger.info("Starting product matching with Estonian NLP")

        estonian_nlp.initialize_from_data(db_session)

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
                    # Проверяем, что этот product_id еще не использован для другого товара
                    existing_product = db_session.query(schemas.Product).filter(
                        schemas.Product.product_id == ean_match.product_id
                    ).first()
                    
                    if existing_product:
                        first_candidate.product_id = ean_match.product_id
                        first_candidate.last_matched = func.now()
                        ean_matched_count += 1
                        db_session.commit()
                        logger.info(f"EAN match found: {first_candidate.store_product_name} → {ean_match.store_product_name} (EAN: {first_candidate.ean})")
                        
                        if not existing_product.barcode:
                            existing_product.barcode = first_candidate.ean
                            db_session.commit()
                            logger.info(f"Updated product barcode: {existing_product.product_id} → {first_candidate.ean}")
                
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
        
        # Отфильтровываем продукты, которые уже имеют product_id
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
        
        # Словарь для отслеживания уже сопоставленных продуктов
        processed_products = {}

        for first_candidate in unmatched_products:
            if not first_candidate.store_product_name or "SKIP" in first_candidate.store_product_name:
                continue
                
            # Создаем безопасный уникальный идентификатор для продукта
            # Используем только гарантированно существующие атрибуты
            candidate_id = f"{first_candidate.store_id}_{hash(first_candidate.store_product_name)}"
                
            # Проверяем, не обработан ли уже этот продукт
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
            
            # Фильтруем продукты, которые уже обработаны
            filtered_similar_products = []
            for p in similar_products:
                # Создаем идентификатор для каждого похожего продукта тем же способом
                p_id = f"{p.store_id}_{hash(p.store_product_name)}"
                
                if p_id not in processed_products:
                    filtered_similar_products.append(p)
            
            similar_products = filtered_similar_products

            first_brand, first_main_words, first_taste, first_attributes, first_type_indicators = estonian_nlp.extract_features(first_candidate.store_product_name)
            
            best_match = None
            best_score = 0.0
            best_metrics = {}

            for second_candidate in similar_products:
                second_brand, second_main_words, second_taste, second_attributes, second_type_indicators = estonian_nlp.extract_features(second_candidate.store_product_name)
                
                # Сравниваем продукты с обновленным методом
                nlp_similarity = estonian_nlp.compare_products(
                    first_candidate.store_product_name, 
                    second_candidate.store_product_name
                )
                
                # Сохраняем упрощенные метрики только для лучшего совпадения
                if nlp_similarity > best_score:
                    best_match = second_candidate
                    best_score = nlp_similarity
                    
                    # Дополнительный анализ для заполнения метрик
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
                
                # Логируем и сохраняем в файл все совпадения выше 50%
                if nlp_similarity >= 50.0:
                    # Дополнительный анализ для заполнения метрик
                    word_similarity = estonian_nlp._improved_word_similarity(first_main_words, second_main_words) * 100
                    attribute_score = estonian_nlp._attribute_similarity(first_attributes, second_attributes) * 100
                    unique_words_score = estonian_nlp._simple_unique_words(
                        first_candidate.store_product_name, 
                        second_candidate.store_product_name
                    ) * 100
                    
                    detailed_info = {
                        'word_similarity': word_similarity,
                        'attribute_score': attribute_score,
                        'unique_words_score': unique_words_score,
                        'first_main_words': first_main_words,
                        'second_main_words': second_main_words,
                        'first_attributes': first_attributes,
                        'second_attributes': second_attributes
                    }
                    
                    # Логируем все совпадения выше 50%
                    logger.info(f"Потенциальное совпадение (оценка {nlp_similarity:.1f}%): {first_candidate.store_product_name} → {second_candidate.store_product_name}")
                    
                    # Сохраняем в Excel файл
                    save_potential_matches_to_file(
                        first_candidate.store_product_name,
                        second_candidate.store_product_name,
                        nlp_similarity,
                        detailed_info
                    )
            
            match_threshold = 82.0
            
            product_category = getattr(first_candidate, "category_id", None)
            if product_category and product_category in category_thresholds:
                match_threshold = category_thresholds[product_category]['threshold']
            
            if best_match and best_score >= 80.0:
                logger.info(f"NLP match found for {first_candidate.store_product_name} and {best_match.store_product_name} with score {best_score:.1f}%")
                logger.info(f"Detailed metrics: {best_metrics}")

                nlp_features = {
                    "brand": first_brand,
                    "main_words": first_main_words,
                    "taste": first_taste,
                    "attributes": first_attributes
                }
                
                if not first_candidate.additional_attributes:
                    first_candidate.additional_attributes = {}
                first_candidate.additional_attributes["nlp_features"] = nlp_features

                # Отмечаем оба продукта как обработанные
                processed_products[candidate_id] = True
                
                # Создаем идентификатор для best_match тем же способом
                best_match_id = f"{best_match.store_id}_{hash(best_match.store_product_name)}"
                processed_products[best_match_id] = True

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
                # Отмечаем продукт как обработанный
                processed_products[candidate_id] = True
                
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

