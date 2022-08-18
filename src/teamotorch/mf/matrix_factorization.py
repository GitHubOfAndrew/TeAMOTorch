# FILE CONTAINS:
# MATRIX FACTORIZATION MODEL OBJECT

import torch
import timeit as t

from .embedding_graphs import *
from .predict_graphs import *
from .loss_graphs import *


class MatrixFactorization:

    def __init__(self, n_components, user_repr_graph=LinearEmbedding(), item_repr_graph=LinearEmbedding(),
                 loss_graph=MSELoss(), pred_graph=DotProduct()):
        self.n_components = n_components
        self.user_repr_graph = user_repr_graph
        self.item_repr_graph = item_repr_graph
        self.loss_graph = loss_graph
        self.pred_graph = pred_graph

        self.user_weight = None
        self.item_weight = None

    def feedforward(self):
        # FEEDFORWARD SHOULD ALWAYS BE DOT PRODUCT PREDICTION
        return torch.matmul(self.user_repr, torch.transpose(self.item_repr, 0, 1))

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

            # update embeddings in every epoch
            if isinstance(self.loss_graph, MSELoss):
                user_repr, _ = self.user_repr_graph.get_repr(user_features, self.n_components, self.user_weight[0])
                item_repr, _ = self.item_repr_graph.get_repr(item_features, self.n_components, self.item_weight[0])

            prediction = torch.matmul(user_repr, torch.transpose(item_repr, 0, 1))
            loss = self.loss_graph.get_loss(prediction, torch_train)

            # compute gradient in place
            external_loss_grad = torch.ones(loss.shape)
            loss.backward(gradient=external_loss_grad)

            # optimization step
            optimizer.step()
            end = t.default_timer()

            cumulative_time += (end - start)
            if (epoch + 1) % 25 == 0:
                print(f'Epoch {epoch + 1} | Loss {torch.sum(loss)} | Runtime {cumulative_time:.5f} s')

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