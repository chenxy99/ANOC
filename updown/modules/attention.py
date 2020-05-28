from functools import lru_cache
from typing import Optional

import torch
from torch import nn
from allennlp.nn.util import masked_softmax


class BottomUpTopDownAttention(nn.Module):
    r"""
    A PyTorch module to compute bottom-up top-down attention
    (`Anderson et al. 2017 <https://arxiv.org/abs/1707.07998>`_). Used in
    :class:`~updown.modules.updown_cell.UpDownCell`

    Parameters
    ----------
    query_size: int
        Size of the query vector, typically the output of Attention LSTM in
        :class:`~updown.modules.updown_cell.UpDownCell`.
    image_feature_size: int
        Size of the bottom-up image features.
    projection_size: int
        Size of the projected image and textual features before computing bottom-up top-down
        attention weights.
    """

    def __init__(self, query_size: int, image_feature_size: int, projection_size: int):
        super().__init__()

        self._query_vector_projection_layer = nn.Linear(query_size, projection_size, bias=False)
        self._image_features_projection_layer = nn.Linear(
            image_feature_size, projection_size, bias=False
        )
        self._attention_layer = nn.Linear(projection_size, 1, bias=False)

    def forward(
        self,
        query_vector: torch.Tensor,
        image_features: torch.Tensor,
        image_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Compute attention weights over image features by applying bottom-up top-down attention
        over image features, using the query vector. Query vector is typically the output of
        attention LSTM in :class:`~updown.modules.updown_cell.UpDownCell`. Both image features
        and query vectors are first projected to a common dimension, that is ``projection_size``.

        Parameters
        ----------
        query_vector: torch.Tensor
            A tensor of shape ``(batch_size, query_size)`` used for attending the image features.
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.
        image_features_mask: torch.Tensor
            A mask over image features if ``num_boxes`` are different for each instance. Elements
            where mask is zero are not attended over.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, num_boxes)`` containing attention weights for each
            image features of each instance in the batch. If ``image_features_mask`` is provided
            (for adaptive features), then weights where the mask is zero, would be zero.
        """

        # shape: (batch_size, projection_size)
        projected_query_vector = self._query_vector_projection_layer(query_vector)

        # Image features are projected by a method call, which is decorated using LRU cache, to
        # save some computation. Refer method docstring.
        # shape: (batch_size, num_boxes, projection_size)
        projected_image_features = self._project_image_features(image_features)

        # Broadcast query_vector as image_features for addition.
        # shape: (batch_size, num_boxes, projection_size)
        projected_query_vector = projected_query_vector.unsqueeze(1).repeat(
            1, projected_image_features.size(1), 1
        )

        # shape: (batch_size, num_boxes, 1)
        attention_logits = self._attention_layer(
            torch.tanh(projected_query_vector + projected_image_features)
        )

        # shape: (batch_size, num_boxes)
        attention_logits = attention_logits.squeeze(-1)

        # `\alpha`s as importance weights for boxes (rows) in the `image_features`.
        # shape: (batch_size, num_boxes)
        if image_features_mask is not None:
            attention_weights = masked_softmax(attention_logits, image_features_mask, dim=-1)
        else:
            attention_weights = torch.softmax(attention_logits, dim=-1)

        return attention_weights

    @lru_cache(maxsize=10)
    def _project_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        r"""
        Project image features to a common dimension for applying attention.

        Extended Summary
        ----------------
        For a single training/evaluation instance, the image features remain the same from first
        time-step to maximum decoding steps. To keep a clean API, we use LRU cache -- which would
        maintain a cache of last 10 return values because on call signature, and not actually
        execute itself if it is called with the same image features seen at least once in last
        10 calls. This saves some computation.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.

        Returns
        -------
        torch.Tensor
            Projected image features of shape ``(batch_size, num_boxes, image_feature_size)``.
        """

        return self._image_features_projection_layer(image_features)


class BottomUpTopDownSaliencyAttention(nn.Module):
    r"""
    A PyTorch module to compute bottom-up top-down saliency attention similar to
    (`Anderson et al. 2017 <https://arxiv.org/abs/1707.07998>`_). Used in
    :class:`~updown.modules.updown_cell.UpDownCell`

    Parameters
    ----------
    query_size: int
        Size of the query vector, typically the output of Attention LSTM in
        :class:`~updown.modules.updown_cell.UpDownCell`.
    penultimate_feature_size: int
        Size of the bottom-up penultimate features.
    projection_size: int
        Size of the projected image and textual features before computing bottom-up top-down
        attention weights.
    """

    def __init__(self, query_size: int, penultimate_feature_size: int, projection_size: int):
        super().__init__()

        self._query_vector_projection_layer = nn.Linear(query_size, projection_size, bias=False)
        self._penultimate_features_projection_layer = nn.Linear(
            penultimate_feature_size, projection_size, bias=False
        )
        self._attention_layer = nn.Linear(projection_size, 1, bias=False)

    def forward(
        self,
        query_vector: torch.Tensor,
        penultimate_features: torch.Tensor,
        penultimate_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Compute attention weights over penultimate features by applying bottom-up top-down attention
        over penultimate features, using the query vector. Query vector is typically the output of
        attention LSTM in :class:`~updown.modules.updown_cell.UpDownCell`. Both image features
        and query vectors are first projected to a common dimension, that is ``projection_size``.

        Parameters
        ----------
        query_vector: torch.Tensor
            A tensor of shape ``(batch_size, query_size)`` used for attending the image features.
        penultimate_features: torch.Tensor
            A tensor of shape ``(batch_size, height * width, channel)``. They are extracted from
            saliency attentive model in the penultimate layers
        penultimate_features_mask: torch.Tensor
            A mask over image features if ``num_boxes`` are different for each instance. Elements
            where mask is zero are not attended over. In this scenario, we can directly set it as None.
            Because, in this attention module, we do not need it.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, height * width)`` containing attention weights for each
            penultimate features of each instance in the batch. If ``penultimate_features_mask`` is provided
            (for adaptive features), then weights where the mask is zero, would be zero.
        """

        # shape: (batch_size, projection_size)
        projected_query_vector = self._query_vector_projection_layer(query_vector)

        # Image features are projected by a method call, which is decorated using LRU cache, to
        # save some computation. Refer method docstring.
        # shape: (batch_size, height*width, projection_size)
        projected_penultimate_features = self._project_penultimate_features(penultimate_features)

        # Broadcast query_vector as penultimate_features for addition.
        # shape: (batch_size, height*width, projection_size)
        projected_query_vector = projected_query_vector.unsqueeze(1).repeat(
            1, projected_penultimate_features.size(1), 1
        )

        # shape: (batch_size, height*width, 1)
        attention_logits = self._attention_layer(
            torch.tanh(projected_query_vector + projected_penultimate_features)
        )

        # shape: (batch_size, num_boxes)
        attention_logits = attention_logits.squeeze(-1)

        # `\alpha`s as importance weights for boxes (rows) in the `image_features`.
        # shape: (batch_size, num_boxes)
        if penultimate_features_mask is not None:
            attention_weights = masked_softmax(attention_logits, penultimate_features_mask, dim=-1)
        else:
            attention_weights = torch.softmax(attention_logits, dim=-1)

        return attention_weights

    @lru_cache(maxsize=10)
    def _project_penultimate_features(self, penultimate_features: torch.Tensor) -> torch.Tensor:
        r"""
        Project image features to a common dimension for applying attention.

        Extended Summary
        ----------------
        For a single training/evaluation instance, the image features remain the same from first
        time-step to maximum decoding steps. To keep a clean API, we use LRU cache -- which would
        maintain a cache of last 10 return values because on call signature, and not actually
        execute itself if it is called with the same image features seen at least once in last
        10 calls. This saves some computation.

        Parameters
        ----------
        penultimate_features: torch.Tensor
            A tensor of shape ``(batch_size, height * width, channel)``.

        Returns
        -------
        torch.Tensor
            Projected image features of shape ``(batch_size, height*width, projection_size)``.
        """
        return self._penultimate_features_projection_layer(penultimate_features)


class BottomUpTopDownGatingFunciton(nn.Module):
    r"""
    A PyTorch module to compute bottom-up top-down attention
    (`Anderson et al. 2017 <https://arxiv.org/abs/1707.07998>`_). Used in
    :class:`~updown.modules.updown_cell.UpDownCell`

    Parameters
    ----------
    image_feature_size: int
        Size of the bottom-up image features.
    penultimate_features_size: int
        Size of the bottom-up penultimate features.
    embedding_size: int
        Size of the textual features.
    """

    def __init__(self, hidden_size: int, embedding_size: int):
        super().__init__()

        self._hidden_projection_layer_h1 = nn.Linear(hidden_size, 1, bias=True)
        self._hidden_projection_layer_h2 = nn.Linear(hidden_size, 1, bias=False)
        self._hidden_projection_layer_h3 = nn.Linear(hidden_size, 1, bias=False)
        self._embedding_vector_projection_layer = nn.Linear(embedding_size, 1, bias=False)


    def forward(
        self,
        hidden_features_h1: torch.Tensor,
        hidden_features_h2: torch.Tensor,
        hidden_features_h3: torch.Tensor,
        token_embedding: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute attention weights over image features by applying bottom-up top-down attention
        over image features, using the query vector. Query vector is typically the output of
        attention LSTM in :class:`~updown.modules.updown_cell.UpDownCell`. Both image features
        and query vectors are first projected to a common dimension, that is ``projection_size``.

        Parameters
        ----------
        hidden_features: torch.Tensor
            A tensor of shape ``(batch_size, hidden_size)`` used for lstm hidden feature.
        token_embedding: torch.Tensor
            A tensor of shape ``(batch_size, embedding_size)``.
        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, 1)`` containing probability of the gating value.
        """

        # shape: (batch_size, 1)
        projected_hidden_features_h1 = self._hidden_projection_layer_h1(hidden_features_h1)

        # shape: (batch_size, 1)
        projected_hidden_features_h2 = self._hidden_projection_layer_h2(hidden_features_h2)

        # shape: (batch_size, 1)
        projected_hidden_features_h3 = self._hidden_projection_layer_h3(hidden_features_h3)

        # shape: (batch_size, 1)
        projected_embedding_vector = self._embedding_vector_projection_layer(token_embedding)

        # shape: (batch_size, 1)
        gate_scores = projected_hidden_features_h1 + projected_hidden_features_h2 + projected_hidden_features_h3\
                      + projected_embedding_vector

        # shape: (batch_size, 1)
        gate_sigmoid = torch.sigmoid(gate_scores)

        return gate_sigmoid
