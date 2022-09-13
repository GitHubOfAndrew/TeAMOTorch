# FILE CONTAINS:
# OBJECTS FOR THE LOSS FUNCTIONS TO BE USED IN TRAINING THE MATRIX FACTORIZATION MODEL

import torch
import torch.distributions as tdist
import math
import numpy as np
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

        positive_interaction_mask = torch.greater(torch_interactions.values(), 0.0)

        torch_interactions_indices = torch_interactions.indices().transpose(1, 0)

        positive_interaction_indices = torch_interactions_indices[positive_interaction_mask]

        positive_predictions = torch_prediction_serial[positive_interaction_mask]

        mapped_predictions_sample_per_interaction = torch_sample_predictions[positive_interaction_indices.transpose(1, 0)[0]]

        summation_term = torch.maximum(
            1.0 - positive_predictions.unsqueeze(dim=1) + mapped_predictions_sample_per_interaction, torch.tensor(0.0))
        
        sampled_margin_rank = (n_items / n_samples) * torch.sum(summation_term, dim=1)

        return torch.log(1.0 + sampled_margin_rank)

class KLDivergenceLoss(LossGraph):
    """
    Class representing the Kullback-Leibler Divergence Loss. Models the positive and negative interactions as normal distributions and computes the complement of the CDF of the composite moments.
    """

    def get_loss(self, interactions, torch_prediction_serial, torch_sample_predictions=None, n_items=None, n_samples=None, predictions=None):
        
        # Convert the scipy sparse matrix to torch sparse matrix

        torch_interactions = torch.tensor(interactions.toarray()).to_sparse()

        torch_pos_mask, torch_neg_mask = torch.greater(torch_interactions.values(), 0.0), torch.le(torch_interactions.values(), 0.0)
        
        torch_pos_pred, torch_neg_pred = torch_prediction_serial[torch_pos_mask], torch_prediction_serial[torch_neg_mask]

        torch_pos_var, torch_pos_mean = torch.var_mean(torch_pos_pred, axis=0, unbiased=False)

        torch_neg_var, torch_neg_mean = torch.var_mean(torch_neg_pred, axis=0, unbiased=False)

        overlap_dist = tdist.Normal(torch_neg_mean - torch_pos_mean, torch.sqrt(torch_pos_var+torch_neg_var))

        return 1.0 - overlap_dist.cdf(torch.tensor(0.0))

class WARPLoss(LossGraph):
    """
    Class representing the Weight Approximate Ranking Pairwise Loss (WARP). Compares predictions to one another according to ground truth interactions, will rank each prediction by sampling the items until we find the first violating example.

    Result should be that the item ranks per user are computed to acceptable magnitude.

    NOTE: The WMRB Loss is preferable to the WARP loss as it is far faster, albeit at the
    """

    def get_loss(self, output, interactions, n_users, n_items, torch_sample_predictions=None, torch_prediction_serial=None, n_samples=None):

        # Convert the scipy sparse matrix to torch sparse matrix

        torch_interactions = torch.tensor(interactions.toarray()).to_sparse()

        max_num_trials = torch_interactions.size()[1]-1

        all_labels_idx = torch.arange(torch_interactions.size()[1])

        positive_indices = torch.zeros(output.size()) # [n_users, n_items]
        negative_indices = torch.zeros(output.size())
        L = torch.zeros(output.size()[0])

        for i in range(n_users):

            msk = torch.ones(torch_interactions.size()[1], dtype=torch.bool)

            # Find the positive label for this example
            j = torch_interactions.indices()[1][torch_interactions.indices()[0]==i]

            if j.nelement() == 0:
                continue

            msk[j] = False

            neg_labels_idx = all_labels_idx[msk]

            # initialize the sample_score_margin
            sample_score_margin = -1
            num_trials = 0

            while ((sample_score_margin < 0) and (num_trials < max_num_trials)):
                
                if neg_labels_idx.nelement() == 0:
                    break

                # randomly sample a negative label
                neg_idx = np.random.choice(a=neg_labels_idx, size=1)
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]

                num_trials += 1
                # calculate the score margin 
                sampled_j = np.random.choice(a=j, size=1)
                sample_score_margin = 1 + output[i, neg_idx] - output[i, sampled_j]

            if sample_score_margin < 0:
                # checks if no violating examples have been found 
                continue
            else: 
                loss_weight = np.log(math.floor((n_items-1)/(num_trials)))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1
                positive_indices[i, sampled_j] = 1

        loss = L * (1-torch.sum(positive_indices * output, dim = 1) + torch.sum(negative_indices * output, dim = 1))

        return torch.sum(loss , dim = 0, keepdim = True)


