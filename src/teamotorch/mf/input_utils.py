# FILE CONTAINS:
# UTILITY/HELPER FUNCTIONS TO FACILITATE PROCESSING THE NECESSARY INPUTS INTO OUR MODELS


import torch
from scipy import sparse
import pandas as pd
import numpy as np
import random


def create_iterable_interaction(df):
    map_id_to_user = dict(enumerate(df['User ID'].unique()))
    map_id_to_item = dict(enumerate(df['Items'].unique()))

    map_user_to_id = {val: key for key, val in map_id_to_user.items()}
    map_item_to_id = {val: key for key, val in map_id_to_item.items()}

    n_users, n_items = len(map_user_to_id), len(map_item_to_id)

    df['User ID'] = df['User ID'].apply(lambda x: map_user_to_id[x])

    df['Items'] = df['Items'].apply(lambda x: map_item_to_id[x])

    return df.values.tolist(), n_users, n_items


def mask_train_test_split(interactions, n_users, n_items, test_size=0.2, shuffle=True, return_indices=True):
    """
    Method to train-test split an interaction table in list form:

    the form of interaction should be:

    [[row, col, ratings] for row, col, ratings]

    If using a pandas dataframe, use df.values.tolist() to convert it into this form.

    Unlike typical train-test splitting, this train-test split would preserve the shape of the interactions (so test shape == train shape).

    The train test split will work by masking existing nonzero entries according to the split size.

    :param interactions: python list: a list of rows of the interaction table
    :param n_users: python int: number of users in interactions
    :param n_items: python int: number of items in interactions
    :param test_size: python float: ratio of test samples
    :param shuffle: python bool: whether to shuffle or not (I highly suggest shuffling)
    :param return_indices: python bool: True means you will return indices, False means not
    :return: sparse csr_matrix representing train interactions, likewise for test, and if return_indices = True, then train, test indices
    """
    if shuffle:
        # shuffle in place
        random.shuffle(interactions)
    else:
        pass

    train_size = 1.0 - test_size
    train_thresh = int(train_size * len(interactions))

    train, test = interactions[:train_thresh], interactions[train_thresh:]

    # convert to csr sparse matrix

    train_ratings = [rating for user_id, item_id, rating in train]
    train_row = [user_id for user_id, item_id, rating in train]
    train_col = [item_id for user_id, item_id, rating in train]

    test_ratings = [rating for user_id, item_id, rating in test]
    test_row = [user_id for user_id, item_id, rating in test]
    test_col = [item_id for user_id, item_id, rating in test]

    train_sparse = sparse.csr_matrix((train_ratings, (train_row, train_col)), shape=(n_users, n_items))
    test_sparse = sparse.csr_matrix((test_ratings, (test_row, test_col)), shape=(n_users, n_items))

    if return_indices:
        # return indices and value too
        train_indices = list(zip(zip(train_row, train_col), train_ratings))
        test_indices = list(zip(zip(test_row, test_col), test_ratings))

        return train_sparse, test_sparse, train_indices, test_indices
    else:
        return train_sparse, test_sparse


def test_sparse_transformation(sparse_interactions, li_indices):
    """
    Method to test if sparse interactions are accurate. Tests if the dense representation and the sparse matrix are equivalent.

    :param sparse_interactions: scipy csr_matrix: sparse matrix representing interactions
    :param li_indices: python list: list of indices
    :return: python bool: True indicates our sparse interactions are consistent with what we expect, false is otherwise
    """

    # test for see if the sparse interactions are created properly

    dense_interactions = sparse_interactions.toarray()

    li_int = []
    for tup, val in li_indices:
        row, col = tup
        if dense_interactions[row, col] == val:
            li_int.append(False)

    if not any(li_int):
        return True
    else:
        return False


def df_to_sparse_pipeline(df, test_size=0.2):
    """
    Pipeline to convert pandas dataframe of interactions ---> train-test split scipy sparse csr_matrix objects. Will check if the result is consistent with expected dense representation.

    :param df: pandas dataframe: dataframe containing interaction data
    :param test_size: python float: ratio of test samples
    :return: train, test csr_matrix elements
    """
    # create iterable list of interaction table row, column, ratings corresponding to nonzero entries
    li_df, n_users, n_items = create_iterable_interaction(df)

    # train-test split into sparse matrices of equal size (we mask the entries randomly according to a train-test split ratio)
    train, test, train_indices, test_indices = mask_train_test_split(li_df, n_users, n_items, test_size=0.2,
                                                                     shuffle=True, return_indices=True)

    # check if the sparse representation is correct
    bool_train = test_sparse_transformation(train, train_indices)
    bool_test = test_sparse_transformation(test, test_indices)

    if bool_train and bool_test:
        return train, test
    else:
        print('Please check your input for errors.')
        return


def convert_np_to_torch_sparse(np_arr):
    """
    :param np_arr: a numpy array
    :return: a torch sparse coo matrix instance (i.e. a sparse tensor)

    Purpose: This is a utility function to help convert between a numpy array and a sparse tensor
    """

    tensor_arr = torch.tensor(np_arr, dtype=torch.float32)
    return tensor_arr.to_sparse()


def convert_torch_to_torch_sparse(torch_arr):
    """
    :param torch_arr: a torch tensor
    :return: sparse tensor
    """
    return convert_np_to_torch_sparse(torch_arr.numpy())


def convert_list_to_torch_sparse(li_arr):
    """
    :param li_arr: a python list
    :return: sparse tensor
    """
    return convert_np_to_torch_sparse(np.array(li_arr))


def convert_df_to_torch_sparse(df_arr):
    """
    :param df_arr: pandas dataframe
    :return: sparse tensor
    """
    return convert_np_to_torch_sparse(np.array(df_arr))


def convert_sp_sparse_to_torch_sparse(sp_arr):
    """
    :param sp_arr: a scipy sparse matrix (csr_matrix)
    :return: torch sparse coo tensor
    """

    nonzero_vals = torch.tensor(sp_arr.data, dtype=torch.float32)

    return nonzero_vals.to_sparse()


def convert_to_torch_sparse(arr):
    """
    :param arr: any instance of the following array-like objects: numpy array, list, pandas dataframe, tensorflow tensor, scipy csr_matrix
    :return: tensorflow sparse tensor
    """

    if isinstance(arr, list):
        return convert_list_to_torch_sparse(arr)

    if isinstance(arr, np.ndarray):
        return convert_np_to_torch_sparse(arr)

    if isinstance(arr, pd.DataFrame):
        return convert_df_to_torch_sparse(arr)

    if isinstance(arr, torch.Tensor):
        return convert_torch_to_torch_sparse(arr)

    if isinstance(arr, sparse.csr_matrix):
        return convert_sp_sparse_to_torch_sparse(arr)


def convert_to_tensor_constant(A):
    """
    :param A: instance of an array-like object: numpy arrays, python lists, pandas dataframes/series, tf tensor
    :return: tensorflow tensor
    """

    # NOTE TO OTHER DEVELOPERS: THIS IS MEANT TO BE FLEXIBLE, IF YOU FIND ANY OTHER ARRAY-LIKE DATATYPE,
    # FEEL FREE TO PUT IT IN HERE

    # if input is a tf.Tensor, then don't change it
    if isinstance(A, torch.Tensor):
        pass

    # convert to constant tensor if the datatype is a list or numpy-like array
    if isinstance(A, list) or isinstance(A, np.ndarray) or isinstance(A, pd.core.series.Series):
        return torch.tensor(A, dtype=torch.float32)

    # convert to tensor using convert_to_tensor api if the datatype is a pandas dataframe
    if isinstance(A, pd.DataFrame):
        return torch.tensor(A.to_numpy(), dtype=torch.float32)
