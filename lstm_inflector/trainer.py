import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_

from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, dataset, device):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.device=device

    def train_step(self, inp, inp_mask, target, batch_size, smoothing, grad_clip=None):
        """
        inp: input batch of size: batch_size x seq_len
        target: target batch of size: batch_size x seq_len
        batch_size: the batch_size
        """
        self.model.train()
        self.optimizer.zero_grad()

        enc_out, enc_hiddens = self.model.encode(inp)
        # Initialize hidden states for decoder LSTM
        decoder_hiddens = self.model.init_hiddens(batch_size)
        # Feed in the first decoder input, as a start tag.
        # -> B x 1
        decoder_input = torch.LongTensor(
            [self.dataset.start_idx]
        ).to(self.device).repeat(batch_size).unsqueeze(1)
        loss_func = self.model.get_loss_func(PAD_index=self.dataset.pad_idx, reduction='mean', smooth_param=smoothing)
        # -> seq_len x B x output_size
        preds = torch.empty(target.size(1), target.size(0), self.model.output_size)
        #losses = []
        for t in range(target.size(1)):
            # pred: B x 1 x output_size
            pred, decoder_hiddens = self.model.decode_step(
                decoder_input,
                decoder_hiddens,
                enc_out,
                inp_mask,
                batch_size
            )

            preds[t] = pred.squeeze(1)
            #losses.append(loss_func(pred.squeeze(1), target[:, t]))
            # The next input is the next target (Teacher Forcing)
            # char in the sequence: B x 1
            decoder_input = target[:, t].unsqueeze(1)

        # -> B x output_size x seq_len
        preds = preds.transpose(0, 1).transpose(1, 2)
        loss = loss_func(preds, target)
        #loss = sum(losses) / len(losses)

        loss.backward()
        if grad_clip is not None:
            clip_grad_value_(self.model.parameters(), grad_clip)

        self.optimizer.step()

        return loss
