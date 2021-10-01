import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, device):
        self.device=device

    def evaluate(self, eval_loader, model, print_preds=False):
        model.eval()

        dataset = eval_loader.dataset
        corr_count = 0
        total_count = 0

        for inp, inp_mask, target in tqdm(eval_loader, desc='Evaluating'):
            batch_size = inp.size(0)
            preds = self.predict(dataset, model, inp, inp_mask, batch_size)

            for i, p, t in zip(dataset.decode(inp), dataset.decode(preds), dataset.decode(target)):
                # Reduce the strings to just the sequence until the first EOS token.
                try:
                    p = p[:p.index(dataset.END)+1]
                    t = t[:t.index(dataset.END)+1]
                except ValueError:
                    # If no EOS decoded, just use the entire sequence
                    p = p

                if print_preds:
                    print(f"input: {i}, pred: {p}, target: {t}")
                    print(corr_count)
                    print(total_count)

                total_count += 1
                if p == t:
                    corr_count += 1

            # for i in range(preds.size(0)):
            #     # True if all vals (e.g. tokens/chars) in the tensor match
            #     matches[i] = torch.equal(preds[i], target[i])
            #
            # corr_count += torch.sum(matches).item()
            # # -> B x seq_len List
            # preds_strs = dataset.decode(preds)
            # target_strs = dataset.decode(target)

        dev_acc = corr_count / total_count
        return dev_acc

    def predict(self, dataset, model, inp, inp_mask, batch_size):
        enc_out, enc_hiddens = model.encode(inp)
        # Initialize hidden states for decoder LSTM
        decoder_hiddens = model.init_hiddens(batch_size)
        # Feed in the first decoder input, as a start tag.
        # -> batch_size x 1
        decoder_input = torch.LongTensor(
            [dataset.start_idx]
        ).to(self.device).repeat(batch_size).unsqueeze(1)

        preds = []
        for t in range(inp.size(1) * 3):
            pred, decoder_hiddens = model.decode_step(
                decoder_input,
                decoder_hiddens,
                enc_out,
                inp_mask,
                batch_size
            )

            # The next input is the predicted token for this step
            decoder_input = model.get_predicted(pred)
            preds.append(decoder_input)

            # Break when all batches predicted an END token
            if (decoder_input == dataset.end_idx).all():
                break

        # -> B x seq_len
        return torch.stack(preds).transpose(0, 1).squeeze(2)
