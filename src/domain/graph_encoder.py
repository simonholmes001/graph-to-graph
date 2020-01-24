import numpy as np

from src.domain.edge import Edge
from src.domain.message import Message
from src.domain.node import Node


class GraphEncoder:
    def __init__(self):
        self.graph = None
        self.w_graph_node_features = None
        self.w_graph_edge_features = None
        self.w_graph_neighbor_messages = None
        self.u_graph_node_features = None
        self.u_graph_neighbor_messages = None
        self.number_of_graph_nodes = None
        self.number_of_graph_edges = None

    def encode_graph(self, node_features: np.ndarray, edge_features: np.ndarray, time_steps: int) -> np.ndarray:
        messages = self._send_messages(node_features, edge_features, time_steps)
        encoded_graph = self._get_encoded_graph(messages, node_features)
        return encoded_graph

    def _get_encoded_graph(self, messages: np.ndarray, node_features: np.ndarray) -> np.ndarray:
        encoded_graph = np.zeros(node_features.shape)
        for start_node in range(node_features.shape[0]):
            encoded_graph[start_node] += self._encode_node(messages, node_features, start_node)
        return encoded_graph

    def _encode_node(self, messages: np.ndarray, node_features: np.ndarray, start_node: int) -> np.ndarray:
        node_encoding_features = self.u_graph_node_features[start_node].dot(node_features[start_node])
        node_encoding_messages = self.u_graph_neighbor_messages[start_node].dot(np.sum(messages[start_node], axis=0))
        return self._relu(node_encoding_features + node_encoding_messages)

    def _send_messages(self, node_features: np.ndarray, edge_features: np.ndarray,
                       time_steps: int) -> np.ndarray:
        self.number_of_graph_nodes = self.w_graph_node_features.shape[0]
        messages = np.zeros((self.number_of_graph_nodes,
                             self.number_of_graph_nodes,
                             self.w_graph_node_features.shape[2]))
        for step in range(time_steps):
            messages = self._compose_messages_from_nodes_to_targets(node_features, edge_features, messages)
        return messages

    def _compose_messages_from_nodes_to_targets(self, node_features: np.ndarray, edge_features: np.ndarray,
                                                messages: np.ndarray) -> np.ndarray:
        for start_node in range(self.number_of_graph_nodes):
            current_node = self._create_node(node_features, start_node)
            for target_node_index in range(current_node.neighbors_count):
                current_node.set_target(target_node_index)
                current_edge = self._create_edge(edge_features, current_node)
                node_slice = current_node.get_slice_to_target()
                message = self._get_message_inputs(messages, current_node, target_node_index, current_edge)
                message.compose()
                messages[node_slice] = message.value
        return messages

    def _get_message_inputs(self, messages: np.ndarray, current_node, target_node_index: int, current_edge):
        message = self._create_message()
        node_slice = current_node.get_slice_to_target()
        message.node_input = self.w_graph_node_features[node_slice].dot(current_node.features)
        message.edge_input = self.w_graph_edge_features[node_slice].dot(current_edge.features)
        messages_from_the_other_neighbors = self._get_messages_from_all_node_neighbors_except_target(messages,
                                                                                                     current_node,
                                                                                                     target_node_index)
        message.neighbors_input = self.w_graph_neighbor_messages[node_slice].dot(
            messages_from_the_other_neighbors.value)
        return message

    def _get_messages_from_all_node_neighbors_except_target(self,
                                                            messages: np.ndarray,
                                                            current_node,
                                                            target_node_index: int):
        messages_from_the_other_neighbors = self._create_message()
        messages_from_the_other_neighbors.value = np.zeros(current_node.features.shape[0])
        if current_node.neighbors_count > 1:
            neighbors_slice = current_node.get_slice_to_neighbors_without_current_target(target_node_index)
            messages_from_the_other_neighbors.value = self.w_graph_neighbor_messages[neighbors_slice][0].dot(
                messages[neighbors_slice][0])
        return messages_from_the_other_neighbors

    def _create_node(self, node_features: np.ndarray, start_node: int):
        return Node(start_node, node_features, self.graph)

    @staticmethod
    def _relu(vector: np.ndarray) -> np.ndarray:
        return np.maximum(0, vector)

    @staticmethod
    def _create_edge(edge_features: np.ndarray, current_node):
        return Edge(current_node.node_id, current_node.current_target, edge_features)

    @staticmethod
    def _create_message():
        return Message()
