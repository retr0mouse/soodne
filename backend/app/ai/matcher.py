from typing import Tuple
from rapidfuzz import fuzz

def match_products(name1: str, name2: str) -> Tuple[bool, float]:
    similarity = fuzz.token_set_ratio(name1.lower(), name2.lower())
    return (similarity > 80, round(similarity, 2))
