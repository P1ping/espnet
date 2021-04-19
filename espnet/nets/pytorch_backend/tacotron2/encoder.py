#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron2 encoder related modules."""

import six
import math

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, make_non_pad_mask, to_device

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (PositionwiseFeedForward,)

def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


class Encoder(torch.nn.Module):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS Synthesis by Conditioning WaveNet on Mel
    Spectrogram Predictions`_. This is the encoder which converts either a sequence
    of characters or acoustic features into the sequence of hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        input_layer="embed",
        embed_dim=512,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.5,
        padding_idx=0,
        extra_dim=0,
    ):
        """Initialize Tacotron2 encoder module.

        Args:
            idim (int) Dimension of the inputs.
            input_layer (str): Input layer type.
            embed_dim (int, optional) Dimension of character embedding.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual
        self.extra_dim = extra_dim
        assert extra_dim >= 0

        # define network layer modules
        if input_layer == "linear":
            self.embed = torch.nn.Linear(idim, econv_chans)
        elif input_layer == "embed":
            self.embed = torch.nn.Embedding(idim, embed_dim, padding_idx=padding_idx)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in six.moves.range(econv_layers):
                ichans = (
                    embed_dim if layer == 0 and input_layer == "embed" else econv_chans
                )
                if layer == 0:
                    ichans += extra_dim
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None, extras=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequence. Either character ids (B, Tmax)
                or acoustic feature (B, Tmax, idim * encoder_reduction_factor). Padded
                value should be 0.
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states(B, Tmax, eunits).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        xs = self.embed(xs)
        if self.extra_dim > 0:
            xs = torch.cat([xs, extras], dim=-1)
        xs = xs.transpose(1, 2)

        # xs = self.embed(xs).transpose(1, 2)
        # if self.extra_dim > 0:
        #     xs = torch.cat([xs, extras.transpose(1,2)], dim=1)

        if self.convs is not None:
            for i in six.moves.range(len(self.convs)):
                if self.use_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens.cpu(), batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

    def inference(self, x, extra=None):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        xs = x.unsqueeze(0)
        ilens = torch.tensor([x.size(0)])
        extras = extra.unsqueeze(0) if extra is not None else None

        return self.forward(xs, ilens, extras)[0][0]


class CharacterEncoder(torch.nn.Module):
    def __init__(
        self,
        idim,
        pred_into_type,
        into_type_num,
        reduce_character_embedding,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_filts=9,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.2,
        padding_idx=0,
    ):
        super(CharacterEncoder, self).__init__()
        
        # Normal encoder modules
        self.idim = idim
        self.use_residual = use_residual
        econv_chans = eunits
        self.embed = torch.nn.Linear(idim, econv_chans)
        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in six.moves.range(econv_layers):
                ichans = econv_chans
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # For reduction
        self.reduce_character_embedding = reduce_character_embedding
        self.query = None # For embedding reduction
        if reduce_character_embedding or pred_into_type:
            query = torch.nn.Parameter(
                torch.FloatTensor((eunits)),
                requires_grad=True
            )
            self.query = torch.nn.init.uniform_(query)
            self.d_k = math.sqrt(eunits)
            self.K = torch.nn.Linear(eunits, eunits)
            self.V = torch.nn.Linear(eunits, eunits)
            self.score_dropout = torch.nn.Dropout(p=dropout_rate)
        
        # For prediction
        self.pred_prj = None
        if pred_into_type:
            self.pred_prj = torch.nn.Linear(eunits, into_type_num)

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        xs = self.embed(xs).transpose(1, 2)
        if self.convs is not None:
            for i in six.moves.range(len(self.convs)):
                if self.use_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)

        xs_old = xs.transpose(1, 2) # xs after convolution

        if self.blstm is None:
            return xs_old
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(xs_old, ilens.cpu(), batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        if self.query is None:
            return xs, hlens, None
        # Predict sentence type
        # (B, T, 1)
        mask = to_device(xs, make_pad_mask(ilens)).unsqueeze(-1)
        # (B, T, D)
        keys = self.K(xs)
        values = self.V(xs)
        # (B, T, D) -> (B, T, 1)
        logits = torch.sum(keys * self.query, dim=-1).unsqueeze(-1) / self.d_k
        logits = logits.masked_fill(mask, -float('inf'))
        scores = F.softmax(logits, dim=1)
        # (B, T, 1) -> (B, 1, T)
        scores = self.score_dropout(scores.masked_fill(mask, 0.0)).transpose(1, 2)
        # (B, 1, T) * (B, T, D) -> (B, 1, D)
        x = torch.matmul(scores, values)
        # Predict intonation type
        intotype_logits = None
        if self.pred_prj is not None:
            intotype_logits = self.pred_prj(x.squeeze(1))
        # Return repeated squeezed character embeddings or original encoded embedings
        if self.reduce_character_embedding:
            return x.squeeze(1), None, intotype_logits
        
        return xs, hlens, intotype_logits

    def inference(self, x):
        xs = x.unsqueeze(0)
        ilens = torch.tensor([x.size(0)])
        xs, hlens, intotype_logits = self.forward(xs, ilens)
        x = xs[0]
        intotype_logit = None
        if intotype_logits is not None:
            intotype_logit = intotype_logits[0]
        return x, intotype_logit


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        idim,
        pred_into_type,
        into_type_num,
        reduce_character_embedding,
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,
        linear_units=2048,
        num_blocks=3,
        dropout_rate=0.2,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_conv_kernel_size=1,
        padding_idx=-1,
        elayers=None,
        eunits=None,
    ):
        """Construct an Encoder object."""
        super(SentenceEncoder, self).__init__()

        self.conv_subsampling_factor = 1
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(idim, attention_dim),
            torch.nn.LayerNorm(attention_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            pos_enc_class(attention_dim, positional_dropout_rate),
        )
        
        self.normalize_before = normalize_before
        
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        # For reduction
        self.reduce_character_embedding = reduce_character_embedding
        self.query = None # For embedding reduction
        if reduce_character_embedding or pred_into_type:
            query = torch.nn.Parameter(
                torch.FloatTensor((attention_dim)),
                requires_grad=True
            )
            self.query = torch.nn.init.uniform_(query)
            # self.d_k = math.sqrt(eunits)
            self.K = torch.nn.Linear(attention_dim, attention_dim)
            # self.V = torch.nn.Linear(eunits, eunits)
            self.score_dropout = torch.nn.Dropout(p=dropout_rate)
        
        # For prediction
        self.pred_prj = None
        if pred_into_type:
            self.pred_prj = torch.nn.Linear(attention_dim, into_type_num)

    def forward(self, xs, ilens=None):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        masks = to_device(xs, make_non_pad_mask(ilens)).unsqueeze(-2)
        xs = self.embed(xs)
        xs, _ = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.query is None:
            return xs, ilens, None

        # Predict sentence type
        # (B, T, 1)
        mask = to_device(xs, make_pad_mask(ilens)).unsqueeze(-1)
        # (B, T, D)
        keys = torch.tanh(self.K(xs))
        values = xs

        # (B, T, D) -> (B, T, 1)
        logits = torch.sum(keys * self.query, dim=-1).unsqueeze(-1)
        logits = logits.masked_fill(mask, -float('inf'))
        scores = F.softmax(logits, dim=1)
        # (B, T, 1) -> (B, 1, T)
        scores = self.score_dropout(scores.masked_fill(mask, 0.0)).transpose(1, 2)
        # (B, 1, T) * (B, T, D) -> (B, 1, D)
        x = torch.matmul(scores, values)
        # Predict intonation type
        intotype_logits = None
        if self.pred_prj is not None:
            intotype_logits = self.pred_prj(x.squeeze(1))
        # Return repeated squeezed character embeddings or original encoded embedings
        if self.reduce_character_embedding:
            return x.squeeze(1), None, intotype_logits
        
        return xs, ilens, intotype_logits

    def inference(self, x):
        xs = x.unsqueeze(0)
        ilens = torch.tensor([x.size(0)])
        xs, hlens, intotype_logits = self.forward(xs, ilens)
        x = xs[0]
        intotype_logit = None
        if intotype_logits is not None:
            intotype_logit = intotype_logits[0]
        return x, intotype_logit
