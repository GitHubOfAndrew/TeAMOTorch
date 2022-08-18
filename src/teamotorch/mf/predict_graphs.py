# FILE CONTAINS:
# PREDICTION GRAPH OBJECT

import torch
from abc import *


class PredictionGraph(ABC):
    """
    Abstract Base Class serving as a template for all prediction graphs.
    """

    @abstractmethod
    def get_prediction(self, U, V):
        """
        Method to return the prediction using the output of choice.

        :param U: torch tensor: the user embedding
        :param V: torch tensor: the item embedding
        :return: torch tensor: the prediction corresponding to each user
        """
        pass


class DotProduct(PredictionGraph):

    def get_prediction(self, U, V):
        return torch.matmul(U, torch.transpose(V, 0, 1))
