# FILE CONTAINS:
# EMBEDDING OBJECTS

from abc import *
import torch

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
