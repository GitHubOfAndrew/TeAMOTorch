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

class WMRBLoss(LossGraph):

    def get_loss(self, interactions, torch_sample_predictions, torch_prediction_serial, n_items, n_samples, predictions=None):

        # Convert the scipy sparse matrix to torch sparse matrix

        torch_interactions = torch.tensor(interactions.toarray()).to_sparse()

        # this wmrb only takes into account positive interactions, ignores all negative interactions

        positive_interaction_mask = torch.greater(torch_interactions.coalesce().values(), 0.0)

        torch_interactions_indices = torch_interactions.coalesce().indices().transpose(1, 0)

        positive_interaction_indices = torch_interactions_indices[positive_interaction_mask]

        positive_predictions = torch_prediction_serial[positive_interaction_mask]

        mapped_predictions_sample_per_interaction = torch_sample_predictions[positive_interaction_indices.transpose(1, 0)[0]]

        summation_term = torch.maximum(
            1.0 - positive_predictions.unsqueeze(dim=1) + mapped_predictions_sample_per_interaction, torch.tensor(0.0))
        
        sampled_margin_rank = (n_items / n_samples) * torch.sum(summation_term, axis=1)

        return torch.log(1.0 + sampled_margin_rank)

