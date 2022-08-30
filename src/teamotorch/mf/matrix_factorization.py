# FILE CONTAINS:
# MATRIX FACTORIZATION MODEL OBJECT

import torch
import timeit as t

from .embedding_graphs import *
from .predict_graphs import *
from .loss_graphs import *
from .utils import *


class MatrixFactorization:

    def __init__(self, n_components, user_repr_graph=LinearEmbedding(), item_repr_graph=LinearEmbedding(),
                 loss_graph=MSELoss(), pred_graph=DotProduct(), n_users=None, n_items=None, n_samples=None, is_sample_based=False):
        self.n_components = n_components
        self.user_repr_graph = user_repr_graph
        self.item_repr_graph = item_repr_graph
        self.loss_graph = loss_graph
        self.pred_graph = pred_graph

        # check for attributes if sample-based loss is used
        self.n_users = n_users
        self.n_items = n_items
        self.n_samples = n_samples

        if self.n_samples is None and self.n_items is not None:
            self.n_samples = n_items // 10

        if isinstance(self.loss_graph, WMRBLoss):
            # prepare all necessary parameters for sample-based WMRB
            self.is_sample_based=True
            self.sample_indices = random_sampler(self.n_items, self.n_users, self.n_samples)

        self.user_weight = None
        self.item_weight = None

    def feedforward(self):
        # FEEDFORWARD SHOULD ALWAYS BE DOT PRODUCT PREDICTION
        return self.pred_graph.get_prediction(self.user_repr, self.item_repr)

    def fit(self, user_features, item_features, torch_train, epochs=100, lr=1e-3):
        # torch_interaction is a torch tensor

        cumulative_time = 0.0

        # initialize embeddings and weights
        user_repr, user_weight = self.user_repr_graph.get_repr(user_features, self.n_components, self.user_weight)
        item_repr, item_weight = self.item_repr_graph.get_repr(item_features, self.n_components, self.item_weight)

        # store weights
        self.user_weight = user_weight
        self.item_weight = item_weight

        optimizer = torch.optim.Adam(self.user_weight + self.item_weight, lr=lr)

        for epoch in range(epochs):
            start = t.default_timer()

            # clear out all gradients in optimizer from previous run
            optimizer.zero_grad()

            # update embeddings in every epoch
            user_repr, _ = self.user_repr_graph.get_repr(user_features, self.n_components, self.user_weight[0])
            item_repr, _ = self.item_repr_graph.get_repr(item_features, self.n_components, self.item_weight[0])

            prediction = torch.matmul(user_repr, torch.transpose(item_repr, 0, 1))

            if isinstance(self.loss_graph, MSELoss):
                # get loss function for MSE Loss
                loss = self.loss_graph.get_loss(prediction, torch_train)

            if isinstance(self.loss_graph, WMRBLoss):
                # get loss function for WMRB Loss
                torch_prediction_serial = get_predictions_serial(torch_train, prediction)
                torch_sample_predictions = get_sampled_predictions(prediction, self.sample_indices)
                loss = self.loss_graph.get_loss(torch_train, torch_sample_predictions, torch_prediction_serial, self.n_items, self.n_samples)

            # compute gradient in place
            external_loss_grad = torch.ones(loss.shape)
            loss.backward(gradient=external_loss_grad)

            # optimization step
            optimizer.step()
            end = t.default_timer()

            cumulative_time += (end - start)
            if (epoch + 1) % 25 == 0:
                print(f'Epoch {epoch + 1} | Loss {torch.mean(loss)} | Runtime {cumulative_time:.5f} s')

            # store the final value of embeddings
            self.user_repr, self.item_repr = user_repr, item_repr

    def retrieve_user_recs(self, k=None, user=None):
        # return item ranks for given user(row) and k(number of items)

        prediction = self.feedforward()

        num_users, num_items = prediction.shape

        # if user is not specified, but rank is specified, return top k item rankings for all users
        if user is None and k is not None:
            return torch.topk(prediction, k=k, dim=1).indices.detach().numpy()
        # if user is specified, but k is specified, return all rankings for specified user
        if user is not None and k is None:
            return torch.topk(prediction[user], k=num_items, dim=1).indices.detach().numpy()
        # if user is specified and k is specified, return top k rankings for specific user
        if user is not None and k is not None:
            return torch.topk(prediction[user], k=k, dim=1).indices.detach().numpy()
        # if user is not specified and k is not specified, return all item rankings for all users
        if user is None and k is None:
            return torch.topk(prediction, k=num_items, dim=1).indices.detach().numpy()


    def recall_at_k(self, interaction, k=10):
        """
        Computes the recall at k: the proportion of known items in the top k predictions.

        Follow LightFM's implementation of recall @ k

        :param interaction: torch tensor: dense interaction table
        :param k: python int: number of top ranked items to look at
        :return: torch tensor: the recall @ k per user (row)
        """
        pass

    def recall_at_k(self, interaction, k=10, preserve_rows=False):
        """
        Computes the recall at k: the proportion of known items in the top k predictions.

        Follow LightFM's implementation of recall @ k

        :param interaction: torch tensor: dense interaction table
        :param k: python int: number of top ranked items to look at
        :param preserve_rows: python boolean: flag indicating
        :return: torch tensor: the recall @ k per user (row)
        """
        prediction = self.feedforward()

        positive_interactions = torch.where(interaction > 0.0, interaction, torch.tensor(0.0))

        # top k from predictions
        top_k_items = torch.topk(prediction, k=k, dim=1).indices

        # gather items for each user in interaction matrix that correspond to the top k items
        res_top_k = torch.gather(interaction, 1, top_k_items)

        hits = torch.count_nonzero(res_top_k, dim=1).type(torch.float32)

        relevant = torch.count_nonzero(positive_interactions, dim=1).type(torch.float32)

        if not preserve_rows:
            mask_zero_interaction = relevant != 0.0
            masked_hits = torch.masked_select(hits, mask_zero_interaction)
            masked_rel = torch.masked_select(relevant, mask_zero_interaction)
            return masked_hits / masked_rel
        else:
            recall = hits / relevant
            nan_mask = torch.isnan(recall)

            return torch.where(nan_mask == False, recall, torch.tensor(0.0))

    def precision_at_k(self, interaction, k=10, preserve_rows=False):
        """
        Computes the precision at k: the proportion of positive predictions in the top k predictions.

        Follow LightFM's implementation of precision @ k

        :param interaction: torch tensor: dense interaction table
        :param k: python int: number of top ranked items to look at
        :param preserve_rows: python boolean: flag indicating
        :return: torch tensor: the precision @ k per user (row)
        """
        prediction = self.feedforward()

        positive_interactions = torch.where(interaction > 0.0, interaction, torch.tensor(0.0))

        # top k from predictions
        top_k_items = torch.topk(prediction, k=k, dim=1).indices

        # gather items for each user in interaction matrix that correspond to the top k items
        res_top_k = torch.gather(interaction, 1, top_k_items)

        hits = torch.count_nonzero(res_top_k, dim=1).type(torch.float32)

        relevant = torch.count_nonzero(positive_interactions, dim=1).type(torch.float32)

        if not preserve_rows:
            mask_zero_interaction = relevant != 0.0
            masked_hits = torch.masked_select(hits, mask_zero_interaction)
            return masked_hits / k
        else:
            return hits / k

    def f1_at_k(self, interaction, k=10, beta=1.0):
        """
        Returns a beta-f1 score at k. This is just a harmonic mean of the precision at k and recall at k.

        :param interaction: torch tensor: dense interaction table
        :param k: python int: number of top ranked items to look at
        :param beta: parameter indicating how much more to weigh precision than recall
        :return: torch float: the mean f1 at k
        """

        precision, recall = self.precision_at_k(interaction, k=k), self.recall_at_k(interaction, k=k)

        prec, rec = torch.mean(precision), torch.mean(recall)

        return ((1 + beta ** 2) * prec * rec) / (beta ** 2 * (prec + rec))

