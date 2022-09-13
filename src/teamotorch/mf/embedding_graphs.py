# FILE CONTAINS:
# EMBEDDING OBJECTS

from abc import *
import torch
from torch import nn

class Embeddings(ABC):
    """
    Abstract Base Class representing the embedding object to be put into the MatrixFactorization object.
    """

    @abstractmethod
    def get_repr(self, features, n_components, weight=None):
        """
        Method to compute the representation of the input feature in a latent space of dimension n_components

        :param features: torch tensor: the features
        :param n_components: python int: the number of latent dimensions we will embed features into
        :param weight: torch tensor: trainable weight
        :return: torch tensor: the embedding of the input feature, trainable; list containing weights: to be put into the optimizer at training time
        """
        pass


class LinearEmbedding(Embeddings):

    def get_repr(self, features, n_components, weight=None):
        _, n_features = features.shape
        if weight is None:
            weight = torch.normal(mean=0.0, std=1.0, size=(n_features, n_components), requires_grad=True)

        return torch.matmul(features, weight), [weight]

class BiasedLinearEmbedding(Embeddings):

    def get_repr(self, features, weights, linear_bias=None):

        _, n_components = weights.shape

        if linear_bias is None:
            linear_bias = torch.zeros(size=(1, n_components), dtype=torch.float, requires_grad=True)

        # Linear bias will be broadcasted to all rows of the features x weights matrix, serving as a bias per user/item
        return torch.matmul(features, weights) + linear_bias, [weights, linear_bias]

class ReLUEmbedding(Embeddings):

    def get_repr(self, features, weights, aux_dim=None, relu_weight=None, relu_bias=None):
          
        _, n_features = features.shape 

        if aux_dim is None:
            _, n_components = weights.shape
            aux_dim = 5 * n_components 

        # generate the ReLU weights randomly
        if relu_weight is None:
            relu_weight = torch.normal(mean=0.0, std=1.0, size=(n_features, aux_dim), dtype=torch.float, requires_grad=True)
        if relu_bias is None:
            relu_bias = torch.zeros(size=(1, aux_dim), dtype=torch.float, requires_grad=True)

        relu = nn.ReLU()
        relu_output = relu(torch.matmul(features, relu_weight) + relu_bias)

        return torch.matmul(relu_output, weights), [weights, relu_weight, relu_bias]
