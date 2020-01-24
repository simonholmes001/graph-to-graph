import numpy as np


class Edge:
    def __init__(self, current_node: int, current_target: int, features: np.ndarray):
        self.current_node = current_node
        self.current_target = current_target
        self.features = self._extract_edge_features(features)

    def _extract_edge_features(self, features: np.ndarray) -> np.ndarray:
        return features[self.current_node, self.current_target]
