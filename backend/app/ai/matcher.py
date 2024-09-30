from typing import Tuple
import difflib

def match_products(name1: str, name2: str) -> Tuple[bool, float]:
    similarity = difflib.SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    return (similarity > 0.8, round(similarity, 2))