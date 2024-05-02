import random
from typing import TypeVar, List, Tuple

X = TypeVar('X') # Generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    # Split data into fractions [prob, 1 - prob]
    data = data[:] # Shallow copy
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]

# Fraction of correct predictions
def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

# Accuracy of our POSITIVE predictions
def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp/ (tp + fp)

# Measures what fraction of the positives our model identified:
def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fn, fn, tn)
    return 2 * p * r / (p + r)