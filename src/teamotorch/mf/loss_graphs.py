# FILE CONTAINS:
# OBJECTS FOR THE LOSS FUNCTIONS TO BE USED IN TRAINING THE MATRIX FACTORIZATION MODEL

import torch
from abc import *


class LossGraph(ABC):
    """
    Abstract Base Class serving as a template for loss functions.
    """

    @abstractmethod
    def get_loss(self, output, train_sparse):
        """
        Returns the loss per user.

        :param output: torch tensor: represents the output of the feedforward
        :param train_sparse: torch sparse tensor: sparse tensor representing the train data
        :return: torch tensor: contains the loss per user
        """

class MSELoss(LossGraph):

    def get_loss(self, output, train_sparse):
        train_dense = train_sparse.to_dense()

        # get mask for prediction output
        nonzero_mask = (train_dense != 0.0)
        masked_output = torch.masked_select(output, nonzero_mask)

        return torch.pow(masked_output - train_sparse._values(), 2.0) / torch.count_nonzero(train_dense)

