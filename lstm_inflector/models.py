import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import sys

class Attn(nn.Module):
    def __init__(self, encoder_outputs_size, hidden_size):
        super(Attn, self).__init__()
        self.encoder_outputs_size = encoder_outputs_size
        self.hidden_size = hidden_size
        # MLP to run over encoder_outputs
        self.M = nn.Linear(self.encoder_outputs_size +\
                               self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden, encoder_outputs, mask):
        """
        Compute the attention distribution from the encoder outputs.

        hidden: B x decoder_dim
        encoder_outputs: B x seq_len x encoder_dim
        mask: B x seq_len
        """
        # Repeat hidden to be copied for each encoder output
        # -> B  x seq_len x decoder_dim
        H = hidden.repeat(1, encoder_outputs.size(1), 1)
        # Get the scores of each time step in the output
        attn_scores = self.score(H, encoder_outputs)
        # Mask the scores with -inf at each padded character
        # So that softmax computes a 0 towards the distribution
        # For that cell.
        attn_scores.data.masked_fill_(mask, -float('inf'))
        # -> B x 1 x seq_len
        weights = F.softmax(attn_scores, dim=1).unsqueeze(1)
        # -> B x 1 x decoder_dim
        weighted = torch.bmm(weights, encoder_outputs)

        return weighted, weights

    def score(self, hidden, encoder_outputs):
        """
        Compute the scores with concat attention from Luong et al 2015
        """
        # -> B x seq_len x encoder_dim + hidden_dim
        concat = torch.cat([encoder_outputs, hidden], 2)
        # V * feed forward w/ tanh
        # -> B x seq_len x hidden_size
        m = self.M(concat)
        # -> B x seq_len x 1
        scores =  self.V(torch.tanh((m)))
        # return B x seq_len
        return scores.squeeze(2)


class EncoderDecoderAttention(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, bidirectional=True):
        super(EncoderDecoderAttention, self).__init__()
        # Size of the input vocab
        self.input_size = input_size
        # dims of each embedding
        self.embedding_size = embedding_size
        self.bidirectional=bidirectional
        # dims of the hidden state
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        # Initial hidden state whose parameters are shared across all examples
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))
        self.encoder = nn.LSTM(
            self.embedding_size, self.hidden_size, batch_first=True, bidirectional=bidirectional
        )
        enc_size = self.hidden_size * 2 if bidirectional == True else self.hidden_size
        self.decoder = nn.LSTM(
            self.embedding_size + enc_size, self.hidden_size, batch_first=True
        )
        self.attention = Attn(enc_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(
        self, input: torch.Tensor, input_mask: torch.Tensor,
        decoder_input: torch.Tensor, target: torch.Tensor=None
    ):
        raise NotImplementedError

    def encode(self, input: torch.Tensor):
        # Get embeddings over input seq
        # -> B x seq_len x embedding_dim
        embedded = self.embedding(input)
        # -> B x seq_len x encoder_dim, (h0, c0)
        return self.encoder(embedded)

    def decode_step(
        self, input: torch.Tensor, last_hiddens: torch.Tensor, enc_out: torch.Tensor,
        enc_mask: torch.Tensor, batch_size: int
    ):
        """Decode one step

        input: B x 1 - The previously decoded token
        last_hiddens: (1 x B x decoder_dim, 1 x B x decoder_dim) - the last hidden states from the decoder
        enc_out: B x seq_len x encoder_dim - the encoded input sequence
        enc_mask: B x seq_len - the mask for the encoded input batch
        batch_size: The size of the batch (referred to as B here)"""
        embedded = self.embedding(input)
        # -> 1 x B x  decoder_dim
        last_h0, last_c0 = last_hiddens
        context, attn_weights = self.attention(last_h0.transpose(0, 1), enc_out, enc_mask)
        # input to decoder LSTM is the embedding concatenated to the weighted, encoded, inputs.
        output, hiddens = self.decoder(torch.cat((embedded, context), 2), (last_h0, last_c0 ))

        # Classify into output vocab
        # -> B x 1 x output_size
        output = self.classifier(output)
        # Compute log_softmax scores for NLLLoss
        scores = self.log_softmax(output)

        return scores, hiddens

    def get_predicted(self, preds):
        """Pick the best index from the vocabulary

        preds: B x seq_len x vocab_size"""
        vals, indices = torch.max(preds, dim=2)
        return indices

    def init_hiddens(self, batch_size):
        """Initialize the hidden state to pass to the LSTM

        Note we learn the initial state h0 as a parameter of the model."""
        # (seq_len x batch_size x hidden_size, seq_len x batch_size x hidden_size)
        return (
            self.h0.repeat(1, batch_size, 1),
            self.c0.repeat(1, batch_size, 1)
        )

    def get_loss_func(self, PAD_index, reduction, smooth_param):
        # This should return the function itself
        if smooth_param is None:
            return torch.nn.NLLLoss(ignore_index=PAD_index, reduction=reduction)
        else:
            def _smooth_nllloss(predict, target):
                predict = predict.transpose(1,2).reshape(-1, self.output_size)
                target = target.view(-1, 1)
                non_pad_mask = target.ne(PAD_index)
                nll_loss = -predict.gather(dim=-1, index=target)[non_pad_mask].mean()
                smooth_loss = -predict.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
                smooth_loss = smooth_loss / self.output_size
                loss = (1.0 - smooth_param) * nll_loss + smooth_param * smooth_loss

                return loss

            return _smooth_nllloss


class EncoderDecoder(EncoderDecoderAttention):

    def decode_step(
        self, input: torch.Tensor, last_hiddens: torch.Tensor, enc_out: torch.Tensor,
        enc_mask: torch.Tensor, batch_size: int
    ):
        """Decode one step

        input: B x 1 - The previously decoded token
        last_hiddens: (1 x B x decoder_dim, 1 x B x decoder_dim) - the last hidden states from the decoder
        enc_out: B x seq_len x encoder_dim - the encoded input sequence
        enc_mask: B x seq_len - the mask for the encoded input batch
        batch_size: The size of the batch (referred to as B here)"""
        embedded = self.embedding(input)
        # -> 1 x B x  decoder_dim
        last_h0, last_c0 = last_hiddens
        # -> B x 1 x encoder_dim
        last_enc_hidden = enc_out[:, -1, :].unsqueeze(1)
        # input to decoder LSTM is the embedding concatenated to the weighted, encoded, inputs.
        output, hiddens = self.decoder(torch.cat((embedded, last_enc_hidden), 2), (last_h0, last_c0 ))

        # Classify into output vocab
        # -> B x 1 x output_size
        output = self.classifier(output)
        # Compute log_softmax scores for NLLLoss
        scores = self.log_softmax(output)

        return scores, hiddens
