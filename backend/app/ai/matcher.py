from datetime import timedelta, datetime
import re
from sqlalchemy import func
from app.core.logger import setup_logger
from app.models.product_store_data import ProductStoreData
from app.models.unit import Unit
from app import schemas
from app.services import product_service
from collections import Counter
import string
import os
import pandas as pd
from rapidfuzz import fuzz, process
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
        
        # Базовые маркеры вкуса для начального обнаружения
        base_taste_markers = ['maitsega', 'maitse', 'mts', 'maits', 'ga', 'ega', 'maitseline', 
                              'maitsest', 'maits-', '-maits', 'maitsestatud', 'maustatud', 
                              'aroom', 'lõhn', 'maitsel', 'mait.', 'm-ga']
        for marker in base_taste_markers:
            self.taste_markers.append(marker)
        
        # Базовые суффиксы для эстонского языка
        self.estonian_suffixes = ['ga', 'ega', 'ne', 'line', 'st', 'ni', 'dega', 'tatud', 'na', 'tega']
        
        self.update_common_words(product_names)
        self.update_taste_markers(product_names)
        
    def update_taste_markers(self, product_names):
        """Динамически идентифицирует вкусовые маркеры на основе анализа продуктов"""
        if not product_names:
            return
            
        # Словарь для отслеживания потенциальных вкусовых слов и их частоты
        potential_taste_words = {}
        
        # Анализируем названия продуктов для поиска вкусовых шаблонов
        for name in product_names:
            if not name:
                continue
                
            # Преобразуем в нижний регистр для анализа
            name_lower = name.lower()
            
            # Шаг 1: Поиск с использованием базовых маркеров
            for marker in self.taste_markers:
                if marker in name_lower:
                    words = name_lower.split()
                    marker_index = -1
                    
                    # Найти слово с маркером
                    for i, word in enumerate(words):
                        if marker in word:
                            marker_index = i
                            break
                    
                    # Проверить слова до и после маркера
                    for offset in [-1, 1]:  # Проверяем слово до и после
                        neighbor_idx = marker_index + offset
                        if 0 <= neighbor_idx < len(words):
                            neighbor = words[neighbor_idx]
                            if len(neighbor) > 2 and not neighbor.isdigit():
                                potential_taste_words[neighbor] = potential_taste_words.get(neighbor, 0) + 1
            
            # Шаг 2: Поиск словосочетаний со словом "maitse" (вкус)
            flavor_matches = re.findall(r'([a-zA-ZäöüõÄÖÜÕ]+(?:\s+[a-zA-ZäöüõÄÖÜÕ]+)?)\s+maitse[a-zA-ZäöüõÄÖÜÕ]*', name_lower)
            for match in flavor_matches:
                if match.strip() and len(match.strip()) > 2:
                    potential_taste_words[match.strip()] = potential_taste_words.get(match.strip(), 0) + 2  # Усиленный вес
            
            # Шаг 3: Поиск слов с суффиксами, характерными для вкусовых описаний
            words = name_lower.split()
            for word in words:
                for suffix in self.estonian_suffixes:
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:
                        base_word = word[:-len(suffix)]
                        if len(base_word) > 2:
                            potential_taste_words[base_word] = potential_taste_words.get(base_word, 0) + 1
                            potential_taste_words[word] = potential_taste_words.get(word, 0) + 1
        
        # Шаг 4: Добавление часто встречающихся вкусовых слов в список маркеров
        total_products = len(product_names)
        min_frequency = max(3, total_products * 0.01)  # Минимум 3 вхождения или 1% от всех продуктов
        
        new_markers = []
        for word, count in potential_taste_words.items():
            if count >= min_frequency and word not in self.taste_markers:
                new_markers.append(word)
                # Добавляем также слова с распространенными суффиксами
                for suffix in self.estonian_suffixes:
                    suffixed_word = word + suffix
                    if suffixed_word not in self.taste_markers:
                        new_markers.append(suffixed_word)
        
        # Добавляем новые маркеры в основной список
        self.taste_markers.extend(new_markers)
        
        logger.debug(f"Динамически обновлен список вкусовых маркеров. Всего маркеров: {len(self.taste_markers)}")
        
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
        
    def extract_features(self, product_name):
        """Extract main words and attributes from product name with improved logic"""
        if not product_name:
            return [], []
        
        # Нормализация входной строки
        product_name = normalize_string(product_name) if isinstance(product_name, str) else ""
        
        # Удаление знаков препинания и приведение к нижнему регистру
        clean_name = re.sub(r'[.,;:!?"\'()]', ' ', product_name.lower())
        
        # Токенизация
        words = [w for w in re.split(r'\s+', clean_name) if w]
        
        # Отдельное сохранение числовых атрибутов и единиц измерения
        attributes = []
        main_words = []
        flavor_words = []
        
        # Регулярное выражение для определения числовых значений с единицами измерения
        weight_pattern = re.compile(r'^(\d+[.,]?\d*)(g|gr|kg|ml|l|cl|tk|tk\.|gb|tb|x\d+.*)$', re.IGNORECASE)
        percentage_pattern = re.compile(r'^(\d+[.,]?\d*)%$')
        dimension_pattern = re.compile(r'^\d+[xX*]\d+.*$')  # Например, 4x100g
        
        for word in words:
            # Проверка, является ли слово числовым атрибутом
            is_attribute = (
                weight_pattern.match(word) or 
                percentage_pattern.match(word) or
                dimension_pattern.match(word) or
                word.replace(',', '.').replace('x', '').replace('%', '').isdigit()
            )
            
            # Если слово является числовым атрибутом, добавляем его только в атрибуты
            if is_attribute:
                attributes.append(word)
                continue
            
            # Проверяем на наличие вкусовых маркеров
            contains_taste_marker = False
            for marker in self.taste_markers:
                if marker in word and len(word) > len(marker) + 1:
                    contains_taste_marker = True
                    # Добавляем слово как важное для вкуса
                    flavor_words.append(word)
                    
                    # Также добавляем его в основные слова
                    if not self._is_common_word(word) and len(word) > 1:
                        main_words.append(word)
                    break
            
            # Если это не атрибут и не вкусовой маркер - проверяем базовые условия
            if not contains_taste_marker and not self._is_common_word(word) and len(word) > 1:
                main_words.append(word)
        
        # Добавить проверку для разделения составных слов
        final_main_words = []
        for word in main_words:
            # Разделение составных слов типа "maasika-nektariini"
            if '-' in word:
                parts = word.split('-')
                final_main_words.extend([p for p in parts if p and len(p) > 1])
            else:
                final_main_words.append(word)
        
        # Также добавляем отдельно извлеченные вкусовые слова, если они не дублируются
        for word in flavor_words:
            if word not in final_main_words:
                final_main_words.append(word)
        
        return final_main_words, attributes
    
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

    def _calculate_word_similarity(self, words1, words2):
        """Calculate similarity based on words with improved logic"""
        if not words1 or not words2:
            return 0.0
        
        # Создаем множества слов, игнорируя числовые значения
        set1 = set([w for w in words1 if not w.replace('.', '').replace(',', '').isdigit()])
        set2 = set([w for w in words2 if not w.replace('.', '').replace(',', '').isdigit()])
        
        # Если после фильтрации остались пустые множества, возвращаем 0
        if not set1 or not set2:
            return 0.0
        
        # Находим общие слова и уникальные слова для каждого продукта
        common_words = set1.intersection(set2)
        
        # Вычисляем сходство
        similarity = len(common_words) / max(len(set1), len(set2))
        
        return similarity

    def _attribute_similarity(self, attributes1, attributes2):
        """
        Calculate the similarity of two attribute sets.
        Takes into account both exact and partial matches.
        """
        if not attributes1 or not attributes2:
            # If one of the sets is empty, return small value instead of 0
            # If both sets are empty, this can be considered a good match
            if not attributes1 and not attributes2:
                return 1.0
            return 0.3  # Увеличиваем с 0.1 до 0.3 - более мягкий штраф
        
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
        
        # Здесь могут быть дополнительные изменения для улучшения сопоставления атрибутов,
        # но текущий фрагмент кода не включает всю логику _attribute_similarity
        
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
        
        # Проверка на разные ключевые слова (потенциально указывающие на разные вкусы/типы)
        different_key_words = False
        # Слова, которые не совпадают между двумя продуктами
        diff_words1 = set(unique_words1) - set(unique_words2)
        diff_words2 = set(unique_words2) - set(unique_words1)
        
        # Если в обоих наборах есть различающиеся слова
        if diff_words1 and diff_words2:
            # Проверяем, насколько сильно они отличаются
            for word1 in diff_words1:
                for word2 in diff_words2:
                    # Более строгое сравнение для длинных слов
                    if len(word1) > 4 and len(word2) > 4 and self._levenshtein_similarity(word1, word2) < 0.65:
                        different_key_words = True
                        break
                    
        # Если обнаружены серьезные различия, сильно снижаем оценку
        if different_key_words:
            return 0.3  # Раньше мы не делали такую проверку в этой функции
        
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
        
        # Нормализуем продукты перед поиском весов
        norm_product1 = normalize_string(product1.lower() if isinstance(product1, str) else "").replace(',', '.')
        norm_product2 = normalize_string(product2.lower() if isinstance(product2, str) else "").replace(',', '.')
        
        weight1 = re.findall(weight_pattern, norm_product1)
        weight2 = re.findall(weight_pattern, norm_product2)
        
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
        
        # Compare main words
        word_similarity = self._calculate_word_similarity(main_words1, main_words2)
        
        # Если основные слова не совпадают хорошо, прекращаем проверку
        if word_similarity < 0.45:
            return 0.0
        
        # Compare attributes
        attribute_score = self._attribute_similarity(attributes1, attributes2)
        
        # Compare unique words
        unique_words_score = self._simple_unique_words(product1, product2)
        
        # Проверяем разницу составов слов (для определения разных брендов без списка)
        different_words1 = set(main_words1) - set(main_words2)
        different_words2 = set(main_words2) - set(main_words1)
        
        # Проверяем на вкусовые различия с сильным штрафом
        taste_diff_penalty = 1.0
        
        # Проверяем различающиеся слова на наличие вкусовых маркеров
        flavor_diff1 = []
        for word in different_words1:
            word_lower = word.lower()
            for marker in self.taste_markers:
                if (marker in word_lower and len(word) > len(marker) + 1) or word_lower in self.taste_markers:
                    flavor_diff1.append(word)
                    break
        
        flavor_diff2 = []
        for word in different_words2:
            word_lower = word.lower()
            for marker in self.taste_markers:
                if (marker in word_lower and len(word) > len(marker) + 1) or word_lower in self.taste_markers:
                    flavor_diff2.append(word)
                    break
        
        # Увеличенный штраф за различия во вкусе
        if flavor_diff1 or flavor_diff2:
            taste_diff_penalty = 0.5  # 50% штраф за вкусовые различия
        
        # Если есть большая разница в составе слов, снижаем итоговую оценку
        word_diff_penalty = 1.0
        if (different_words1 and different_words2 and 
            len(different_words1) >= 2 and len(different_words2) >= 2):
            word_diff_penalty = 0.7  # 30% штраф за общие различия в словах
        
        # Расчет базовой оценки
        base_score = (
            word_similarity * 0.44 +
            attribute_score * 0.22 +
            unique_words_score * 0.44
        )
        
        # Применяем штрафы последовательно
        total_score = base_score * word_diff_penalty * taste_diff_penalty
        
        # Return percentage for better readability
        return total_score * 100

    def _levenshtein_similarity(self, s1, s2):
        """Calculate string similarity using RapidFuzz library."""
        if not s1 or not s2:
            return 0.0
        
        # Используем ratio из библиотеки RapidFuzz для нормализованного сходства Левенштейна
        # Это возвращает значение между 0 и 100, поэтому делим на 100
        return fuzz.ratio(s1.lower(), s2.lower()) / 100.0

class ProductMatcher:
    def __init__(self, config=None):
        """
        Инициализация матчера с настроенными порогами сопоставления.
        """
        self.config = config or {}
        
        # Установка порогов сопоставления
        # Общий порог для автоматического сопоставления
        self.threshold = 0.85  # Меняем с 0.95 на 0.85 (85%)
        
        # Порог для потенциальных сопоставлений (требуют проверки)
        self.potential_match_threshold = 0.80  # Сопоставления между 85% и 90% требуют проверки
        
        # Пороги для отдельных компонентов
        self.word_similarity_threshold = 0.95  # Порог сходства ключевых слов
        self.attribute_similarity_threshold = 0.95  # Порог сходства атрибутов
        self.unique_words_threshold = 0.75  # Порог сходства уникальных слов

estonian_nlp = EstonianProductNLP()

# Function to save matches to Excel file
def save_potential_matches_to_file(first_product, second_product, score, metrics):
    """Saves information about potential matches to an Excel file with normalized values"""
    try:
        filepath = "potential_matches.xlsx"
        
        # Нормализуем все текстовые значения перед сохранением
        def normalize_if_str(val):
            return normalize_string(val) if isinstance(val, str) else val
        
        # Нормализуем списки
        def normalize_list(lst):
            if isinstance(lst, list):
                return [normalize_if_str(item) for item in lst]
            else:
                return lst
        
        # Находим различающиеся слова
        different_words1 = set(metrics.get('first_main_words', [])) - set(metrics.get('second_main_words', []))
        different_words2 = set(metrics.get('second_main_words', [])) - set(metrics.get('first_main_words', []))
        
        # Проверяем наличие вкусовых маркеров в различающихся словах
        taste_markers = estonian_nlp.taste_markers if hasattr(estonian_nlp, 'taste_markers') else []
        flavor_diff1 = []
        for word in different_words1:
            word_lower = word.lower()
            for marker in taste_markers:
                if (marker in word_lower and len(word) > len(marker) + 1) or word_lower in taste_markers:
                    flavor_diff1.append(word)
                    break
        
        flavor_diff2 = []
        for word in different_words2:
            word_lower = word.lower()
            for marker in taste_markers:
                if (marker in word_lower and len(word) > len(marker) + 1) or word_lower in taste_markers:
                    flavor_diff2.append(word)
                    break
        
        # Проверка на снижение оценки из-за разных составов слов
        has_word_penalty = False
        word_penalty_reason = ""
        if (different_words1 and different_words2 and 
            len(different_words1) >= 2 and len(different_words2) >= 2):
            has_word_penalty = True
            word_penalty_reason = "разные составы ключевых слов"
        
        # Проверка на снижение оценки из-за вкусовых различий
        flavor_penalty = False
        if flavor_diff1 or flavor_diff2:
            flavor_penalty = True
        
        # Рассчитываем скорректированную оценку для проверки
        raw_score = (metrics.get('word_similarity', 0) * 0.44 + 
                    metrics.get('attribute_score', 0) * 0.22 + 
                    metrics.get('unique_words_score', 0) * 0.44)
        
        # Оценка со снижением
        adjusted_score = raw_score
        if has_word_penalty:
            adjusted_score *= 0.7
        
        if flavor_penalty:
            adjusted_score *= 0.5  # Увеличенный штраф за различия во вкусе
        
        # Create data for writing with normalized values and extra info
        data = {
            'First Product': [normalize_if_str(first_product)],
            'Second Product': [normalize_if_str(second_product)],
            'Score': [f"{score:.1f}%"],
            'Raw Score': [f"{raw_score:.1f}%"],
            'Word Similarity': [f"{metrics.get('word_similarity', 0):.1f}%"],
            'Attribute Score': [f"{metrics.get('attribute_score', 0):.1f}%"],
            'Unique Words Score': [f"{metrics.get('unique_words_score', 0):.1f}%"],
            'First Main Words': [str(normalize_list(metrics.get('first_main_words', [])))],
            'Second Main Words': [str(normalize_list(metrics.get('second_main_words', [])))],
            'First Attributes': [str(normalize_list(metrics.get('first_attributes', [])))],
            'Second Attributes': [str(normalize_list(metrics.get('second_attributes', [])))],
            'Different Words First': [str(list(different_words1))],
            'Different Words Second': [str(list(different_words2))],
            'Flavor Differences First': [str(flavor_diff1)],
            'Flavor Differences Second': [str(flavor_diff2)],
            'Has Word Penalty': [str(has_word_penalty)],  # Используем str() вместо прямого булева значения
            'Word Penalty Reason': [word_penalty_reason],
            'Has Flavor Penalty': [str(flavor_penalty)],  # Используем str() вместо прямого булева значения
            'Raw Score': [f"{raw_score:.1f}%"],
            'Adjusted Score': [f"{adjusted_score:.1f}%"],
            'Taste Markers Detected': [str([m for m in taste_markers if m in first_product.lower() or m in second_product.lower()])],
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        
        # Create DataFrame from data
        df = pd.DataFrame(data)
        
        sheet_name = "Matches"
        
        try:
            # Проверяем, существует ли файл и не поврежден ли он
            if os.path.isfile(filepath):
                try:
                    # Пытаемся прочитать существующий файл
                    existing_df = pd.read_excel(filepath, sheet_name=sheet_name)
                    
                    # Если успешно прочитали, добавляем новые данные
                    # Используем .append() вместо ExcelWriter для большей надежности
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    
                    # Записываем объединенный DataFrame
                    combined_df.to_excel(filepath, index=False, sheet_name=sheet_name)
                except Exception as e:
                    logger.warning(f"Could not read existing Excel file, creating new: {str(e)}")
                    # При любой ошибке создаем новый файл
                    df.to_excel(filepath, index=False, sheet_name=sheet_name)
            else:
                # Файл не существует, создаем новый
                df.to_excel(filepath, index=False, sheet_name=sheet_name)
                
            logger.debug(f"Match data successfully saved to {filepath}")
                
        except Exception as e:
            # При любой ошибке создаем файл с временной меткой
            new_filepath = f"potential_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            logger.warning(f"Failed to write to {filepath}, creating new file {new_filepath}: {str(e)}")
            df.to_excel(new_filepath, index=False, sheet_name=sheet_name)
            
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
        
        existing_products = db_session.query(ProductStoreData).filter(ProductStoreData.product_id.isnot(None)).all()
        for product in existing_products:
            product_category = getattr(product, "category_id", None)
            if product_category:
                if product_category not in category_thresholds:
                    category_thresholds[product_category] = {'count': 0, 'threshold': 75.0}
                category_thresholds[product_category]['count'] += 1
        
        # Adjust threshold based on category frequency
        for category_id, data in category_thresholds.items():
            if data['count'] > 50:
                category_thresholds[category_id]['threshold'] = 85.0
            elif data['count'] > 30:
                category_thresholds[category_id]['threshold'] = 83.0
            else:
                category_thresholds[category_id]['threshold'] = 80.0
        
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
                
                # Для метаданных сохраняем только извлеченные признаки
                if nlp_similarity > best_score:
                    best_match = second_candidate
                    best_score = nlp_similarity
                    
                    # Additional analysis to fill metrics
                    word_similarity = estonian_nlp._calculate_word_similarity(first_main_words, second_main_words) * 100
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
            
            match_threshold = 85.0
            
            product_category = getattr(first_candidate, "category_id", None)
            if product_category and product_category in category_thresholds:
                match_threshold = max(category_thresholds[product_category]['threshold'], 85.0)
            
            if best_match and best_score >= match_threshold:
                # Сохраняем только основную информацию без getBrand
                nlp_features = {
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
            if best_score >= 75.0:
                match_metrics["high_confidence_matches"] += 1
                match_metrics["confidence_scores"].append(best_score)
            elif best_score >= 70.0:
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

