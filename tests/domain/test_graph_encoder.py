from unittest import TestCase

import numpy as np

from src.domain.graph_encoder import GraphEncoder
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    BASE_GRAPH_NODES_NUMBER, BASE_GRAPH_EDGES_NUMBER, BASE_W_MATRIX, BASE_U_MATRIX, BASE_GRAPH_EDGE_FEATURES


class TestGraphEncoder(TestCase):

    def setUp(self) -> None:
        self.graph_encoder = GraphEncoder()
        self.graph_encoder.graph = BASE_GRAPH
        self.graph_encoder.w_graph_node_features = 0.1 * BASE_W_MATRIX
        self.graph_encoder.w_graph_edge_features = 0.1 * BASE_W_MATRIX
        self.graph_encoder.w_graph_neighbor_messages = 0.1 * BASE_W_MATRIX
        self.graph_encoder.u_graph_node_features = 0.1 * BASE_U_MATRIX
        self.graph_encoder.u_graph_neighbor_messages = 0.1 * BASE_U_MATRIX
        self.graph_encoder.number_of_graph_nodes = BASE_GRAPH_NODES_NUMBER
        self.graph_encoder.number_of_graph_edges = BASE_GRAPH_EDGES_NUMBER

    def test_encode_graph_returns_the_expected_encoding_for_a_node_after_one_time_step(self):
        # Given
        time_steps = 1
        node = 0
        node_encoding_expected = np.array([0.53, 0.53])

        # When
        node_encoding = self.graph_encoder.encode_graph(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH_EDGE_FEATURES, time_steps)[
            node]

        # Then
        self.assertTrue(np.allclose(node_encoding_expected, node_encoding))

    def test_encode_graph_returns_the_expected_shape(self):
        # Given
        time_steps = 1
        encoded_graph_shape_expected = BASE_GRAPH_NODE_FEATURES.shape

        # When
        encoded_graph_shape = self.graph_encoder.encode_graph(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH_EDGE_FEATURES,
                                                              time_steps).shape

        # Then
        self.assertTrue(np.allclose(encoded_graph_shape_expected, encoded_graph_shape))

    def test_get_the_messages_matrix_with_same_dimensions_as_the_graph(self):
        # Given
        messages_expected = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                      BASE_GRAPH_NODES_NUMBER,
                                      BASE_GRAPH_NODE_FEATURES.shape[1]))

        # When
        messages = self.graph_encoder._compose_messages_from_nodes_to_targets(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH_EDGE_FEATURES,
                                                                              messages_expected)

        # Then
        self.assertTrue(np.array_equal(messages_expected.shape, messages.shape))

    def test_get_the_messages_matrix_with_the_same_links_as_the_graph(self):
        # Given
        messages_initial = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODE_FEATURES.shape[1]))
        messages_non_zero_expected = np.nonzero(BASE_GRAPH)[1]

        # When
        messages = self.graph_encoder._compose_messages_from_nodes_to_targets(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH_EDGE_FEATURES,
                                                                              messages_initial)
        messages_non_zero = np.nonzero(np.sum(messages, axis=2))[1]

        # Then
        self.assertTrue(np.array_equal(messages_non_zero_expected, messages_non_zero))

    def test_get_the_expected_messages_from_the_a_node_after_one_time_step(self):
        messages_initial = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODE_FEATURES.shape[1]))
        node_expected = 0
        messages_from_node_expected = np.array([[0., 0.],
                                                [0.6, 0.6],
                                                [0.55, 0.55],
                                                [0., 0.]])

        # When
        messages_from_node = self.graph_encoder._compose_messages_from_nodes_to_targets(BASE_GRAPH_NODE_FEATURES,
                                                                                        BASE_GRAPH_EDGE_FEATURES,
                                                                                        messages_initial)[node_expected]

        # Then
        self.assertTrue(np.allclose(messages_from_node_expected, messages_from_node))
