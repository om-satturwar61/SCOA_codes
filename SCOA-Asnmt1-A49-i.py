import numpy as np
from time import sleep

class FuzzySet:
    def __init__(self, elements, memberships):
        self.elements = np.array(elements)
        self.memberships = np.array(memberships)
        
        if len(elements) != len(memberships):
            raise ValueError("Elements and memberships must have the same length.")

    def __repr__(self):
        return f"FuzzySet(elements={self.elements}, memberships={self.memberships})"
    
    def complement(self):
        comp_memberships = 1 - self.memberships
        return FuzzySet(self.elements, comp_memberships)
    
    def union(self, other):
        if not np.array_equal(self.elements, other.elements):
            raise ValueError("Fuzzy sets must have the same elements.")
        
        union_memberships = np.maximum(self.memberships, other.memberships)
        return FuzzySet(self.elements, union_memberships)
    
    def intersection(self, other):
        if not np.array_equal(self.elements, other.elements):
            raise ValueError("Fuzzy sets must have the same elements.")
        
        intersection_memberships = np.minimum(self.memberships, other.memberships)
        return FuzzySet(self.elements, intersection_memberships)

elements = ['x1', 'x2', 'x3', 'x4']
memberships_A = [0.2, 0.7, 0.4, 0.9]
A = FuzzySet(elements, memberships_A)
print(f"Fuzzy Set A: {A}")

memberships_B = [0.5, 0.3, 0.8, 0.6]
B = FuzzySet(elements, memberships_B)
print(f"Fuzzy Set B: {B}")

A_complement = A.complement()
print(f"Complement of A: {A_complement}")

A_union_B = A.union(B)
print(f"Union of A and B: {A_union_B}")

A_intersection_B = A.intersection(B)
print(f"Intersection of A and B: {A_intersection_B}")
sleep(100)