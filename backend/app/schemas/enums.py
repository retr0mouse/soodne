from enum import Enum

class MatchingStatusEnum(str, Enum):
    unmatched = "unmatched"
    matched = "matched"
    pending = "pending"
