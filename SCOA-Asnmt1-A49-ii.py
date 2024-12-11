import numpy as np
from time import sleep

class FuzzyRelation:
    def __init__(self, relation):
        self.relation = np.array(relation)
    
    def __repr__(self):
        return f"FuzzyRelation(relation=\n{self.relation})"
    
    def max_min_composition(self, other):
        if self.relation.shape[1] != other.relation.shape[0]:
            raise ValueError("Incompatible fuzzy relations for composition.")
        
        rows_A, cols_A = self.relation.shape
        rows_B, cols_B = other.relation.shape
        result = np.zeros((rows_A, cols_B))
        
        for i in range(rows_A):
            for j in range(cols_B):
                result[i, j] = np.max(np.minimum(self.relation[i], other.relation[:, j]))
        
        return FuzzyRelation(result)

R1 = FuzzyRelation([[0.5, 0.7, 0.2],
                    [0.3, 0.6, 0.9]])
print(f"Fuzzy Relation R1:\n{R1}")

R2 = FuzzyRelation([[0.8, 0.4],
                    [0.1, 0.9],
                    [0.5, 0.3]])
print(f"Fuzzy Relation R2:\n{R2}")

# Perform max-min composition
R_composed = R1.max_min_composition(R2)
print(f"Composed Fuzzy Relation R1 ∘ R2:\n{R_composed}")
sleep(100)
