import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from tqdm import tqdm

import click
import os

from lstm_inflector.models import EncoderDecoderAttention, EncoderDecoder
from lstm_inflector.trainer import Trainer
from lstm_inflector.evaluator import Evaluator
from lstm_inflector import datasets


def get_dataset(dataset: str):
    dataset_fac = {
        "sigmorphon_data": datasets.SigmorphonData,
        "sigmorphon_wug_data": datasets.SigmorphonWugData,
        "sigmorphon_orth_wug_data": datasets.SigmorphonOrthWugData
    }

    try:
        return dataset_fac[dataset]
    except KeyError as e:
        raise Exception(f"")


def get_device(use_gpu):
    torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


@click.command()
@click.option("--train_data_path", type=str, required=True)
@click.option("--dev_data_path", type=str, required=True)
@click.option("--output_path", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--lang", type=str, required=True)
@click.option("--num_epochs", type=int, default=20)
@click.option("--patience", type=int)
@click.option("--learning_rate", type=float, default=.001)
@click.option("--smoothing", type=float, default=None)
@click.option("--gradient_clip", type=float, default=None)
@click.option("--batch_size", type=int, default=16)
@click.option("--eval_batch_size", type=int, default=1)
@click.option("--embedding_size", type=int, default=128)
@click.option("--hidden_size", type=int, default=256)
@click.option("--bidirectional/--no-bidirectional", type=bool, default=True)
@click.option("--attn/--no-attn", type=bool, default=True)
@click.option("--eval_every", type=int, default=1,
                help="How many epochs to train between evaluation on dev")
@click.option("--eval_after", type=int, default=0,
                help="How many epochs to train before starting evaluation on dev")
@click.option('--amsgrad/--no-amsgrad', default=False)
@click.option('--saveall/--no-saveall', default=False)
@click.option('--print-preds/--no-print-preds', default=False)
@click.option('--gpu/--no-gpu', default=True,
                help="whether to use amsgrad variant of Adam optmization")
def main(
    train_data_path, dev_data_path, output_path, dataset, lang, num_epochs, patience,
    learning_rate, smoothing, gradient_clip, batch_size, eval_batch_size,
    embedding_size, hidden_size, bidirectional, attn, eval_every, eval_after,
    amsgrad, saveall, print_preds, gpu
):
    os.makedirs(output_path, exist_ok=True)
    device = get_device(gpu)

    print("Building train dataset")
    dataset_cls = get_dataset(dataset)
    train_set = dataset_cls(train_data_path)
    print("Loaded vocab:")
    print(train_set.char2i)
    train_set.write_index(output_path, lang)
    train_loader = DataLoader(
        train_set, collate_fn=datasets.PadCollate(pad_idx=train_set.pad_idx),
        batch_size=batch_size, shuffle=True
    )
    eval_set = dataset_cls(dev_data_path)
    eval_set.load_index(output_path, lang)
    eval_loader = DataLoader(
        eval_set, collate_fn=datasets.PadCollate(pad_idx=eval_set.pad_idx),
        batch_size=eval_batch_size, shuffle=False
    )

    print(f"Bidirecitnal: {bidirectional}, Attention: {attn}")
    if attn:
        model = EncoderDecoderAttention(
            input_size=train_set.vocab_size, embedding_size=embedding_size,
            hidden_size=hidden_size, output_size=train_set.vocab_size, bidirectional=bidirectional
        ).to(device)
    else:
        model = EncoderDecoder(
            input_size=train_set.vocab_size, embedding_size=embedding_size,
            hidden_size=hidden_size, output_size=train_set.vocab_size, bidirectional=bidirectional
        ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad)
    #optimizer = SGD(model.parameters(), lr=learning_rate)
    trainer = Trainer(
        model=model, optimizer=optimizer, dataset=train_loader.dataset, device=device
    )

    for batch_in, batch_in_mask, batch_target in train_loader:
        print(f"example TRAIN sample")
        for b, t in zip(train_set.decode(batch_in), train_set.decode(batch_target)):
            print(f"INPUT: {b} OUTPUT: {t}")

        print(f"MASK: {batch_in_mask}")

        break

    for batch_in, batch_in_mask, batch_target in eval_loader:
        print(f"example EVAL sample")
        for b, t in zip(eval_set.decode(batch_in), eval_set.decode(batch_target)):
            print(f"INPUT: {b} OUTPUT: {t}")

        print(f"MASK: {batch_in_mask}")

        break

    evaluator = Evaluator(device=device)
    best_acc = -1000
    epochs_since_best = 0
    # Training loop over epochs
    for i in range(num_epochs):
        print(f"EPOCH: {i}")
        batch_losses = []
        for batch_in, batch_in_mask, batch_target in tqdm(train_loader, desc='Training'):
            # Final batch is remainder, so can be variable len
            batch_size = batch_in.size(0)
            batch_loss = trainer.train_step(
                batch_in, batch_in_mask, batch_target, batch_size, smoothing, gradient_clip
            )
            batch_losses.append(batch_loss)

        print(f"Average loss for epoch {i}: {sum(batch_losses)/len(batch_losses)}")

        # Evaluate on dev set
        if i > (eval_after - 1) and (i+1) % eval_every == 0:
            dev_acc = evaluator.evaluate(eval_loader, model, print_preds)
            print(f"Dev accuracy: {dev_acc}")
            if saveall or dev_acc > best_acc:
                model_fn = f"{output_path}/{lang}_{i}_{dev_acc}.pt"
                print(f"Saving model to {model_fn}")
                torch.save(model, model_fn)
            if dev_acc > best_acc:
                epochs_since_best = 0
                best_acc = dev_acc
            else:
                epochs_since_best += 1

            if patience is not None and epochs_since_best > patience:
                break


if __name__=='__main__':
    main()
