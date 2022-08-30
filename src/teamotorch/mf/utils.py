# FILE CONTAIN:
# UTILITY/HELPER FUNCTIONS TO ACCOMPLISH VARIOUS THINGS

import torch
from scipy import sparse
import numpy as np


def generate_random_interaction(n_users, n_items, min_val=0.0, max_val=5.0, density=0.50):
    """
    Generates a random matrix of shape (n_users, n_items) in the range of [min_val, max_val] with density being the proportion of nonzero entries in the tensor.

    :param n_users: python int: number of users (queries)
    :param n_items: python int: number of items (keys)
    :param min_val: python int: minimum value of entries
    :param max_val: python int: maximum value of entries
    :param density: python float: the proportion of nonzero entries in the entire matrix
    :return: torch.sparse.coo_matrix, a sparse torch tensor in coo format
    """

    p = sparse.random(n_users, n_items, density=density)

    p = (max_val - min_val) * p + min_val * p.ceil()

    random_arr = np.round(p.toarray())

    scipy_random_arr = sparse.coo_matrix(random_arr)

    values = scipy_random_arr.data

    indices = np.vstack((scipy_random_arr.row, scipy_random_arr.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scipy_random_arr.shape

    return torch.sparse_coo_tensor(i, v, size=shape)

def random_sampler(n_items, n_users, n_samples, replace=False):

    """
    Generates the sampled column indices (items) per row (user). This should be a matrix of shape [n_users, n_samples]
    :param n_items: python int: number of items in interactions
    :param n_users: python int: number of users in interactions
    :param n_samples: python int: number of item samples
    :param replace: python boolean: whether to sample items by replacement or not
    :return: tensorflow tensor: contains n_samples sampled items per user
    """

    items_per_user = [np.random.choice(a=n_items, size=n_samples, replace=replace) for _ in range(n_users)]

    return torch.tensor(np.array(items_per_user), dtype=torch.int64)

def get_sampled_predictions(predictions, sampled_indices):
    """
    Get the sampled predictions according to the random sampling from the random_sampler() method

    :param predictions:
    :param sampled_indices:
    :return:
    """
    return torch.gather(predictions, 1, sampled_indices)

def get_predictions_serial(interactions, predictions):
    nonzero_mask = torch.tensor(interactions.toarray() != 0)
    return torch.masked_select(predictions, nonzero_mask)




