# FILE CONTAINS:
# OBJECTS FOR THE LOSS FUNCTIONS TO BE USED IN TRAINING THE MATRIX FACTORIZATION MODEL

import torch
from abc import *


class LossGraph(ABC):
    """
    Abstract Base Class serving as a template for loss functions.
    """

    @abstractmethod
    def get_loss(self, output, interactions, torch_sample_predictions, torch_prediction_serial, n_items, n_samples):
        """
        Returns the loss per user.

        :param output: torch tensor: represents the output of the feedforward
        :param interactions: scipy sparse matrix: represents the interaction table in sparse csr format
        :param torch_sample_predictions: torch tensor: represents the prediction output corresponding to the sampled items
        :param torch_prediction_serial: torch tensor: the predictions corresponding to the known interactions
        :param n_items: python int: number of items
        :param n_samples: python int: number of sampled items
        :return: torch tensor: contains the loss per user
        """
        pass


class MSELoss(LossGraph):
    """
    Class representing the Mean Square Error Loss. This only takes into account the known interactions.
    """

    def get_loss(self, output, interactions, torch_sample_predictions=None, torch_prediction_serial=None, n_items=None, n_samples=None):
        train_sparse = torch.tensor(interactions.toarray()).to_sparse()
        train_dense = train_sparse.to_dense()

        # get mask for prediction output
        nonzero_mask = (train_dense != 0.0)
        masked_output = torch.masked_select(output, nonzero_mask)

        return torch.pow(masked_output - train_sparse._values(), 2.0)


class WMRBLoss(LossGraph):
    """
    Class representing the Weighted Margin Rank Batch Loss. Compares predictions corresponding to known interactions and randomly sampled predictions in order to optimize rank.
    """

    def get_loss(self, interactions, torch_sample_predictions, torch_prediction_serial, n_items, n_samples, output=None):

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
        
        sampled_margin_rank = (n_items / n_samples) * torch.sum(summation_term, dim=1)

        return torch.log(1.0 + sampled_margin_rank)


