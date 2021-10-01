import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import click
import os

from lstm_inflector.models import EncoderDecoderAttention
from lstm_inflector.trainer import Trainer
from lstm_inflector.evaluator import Evaluator
from lstm_inflector import datasets


def get_dataset(dataset: str):
    dataset_fac = {
        "sigmorphon_data": datasets.SigmorphonData,
        "sigmorphon_wug_data": datasets.SigmorphonWugData
    }

    try:
        return dataset_fac[dataset]
    except KeyError as e:
        raise Exception(f"")


def get_device():
    torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("--data_path", type=str, required=True)
@click.option("--lang", type=str, required=True)
@click.option("--output_path", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--model_path", type=str, required=True)
@click.option("--data_dir", type=str, required=True)
def main(
    data_path, lang, output_path, dataset, model_path, data_dir
):
    os.makedirs(output_path, exist_ok=True)
    device = get_device()
    batch_size = 1

    print(f"Loading dataset from {data_path}")
    dataset_cls = get_dataset(dataset)
    data_set = dataset_cls(data_path)
    data_set.load_index(data_dir, lang)
    print("Loaded vocab:")
    print(data_set.char2i)
    data_loader = DataLoader(
        data_set, collate_fn=datasets.PadCollate(pad_idx=data_set.pad_idx),
        batch_size=batch_size, shuffle=False
    )

    model = torch.load(model_path).to(device)
    evaluator = Evaluator(device=device)
    batch_losses = []
    for batch_in, batch_in_mask, batch_target in data_loader:
        # Final batch is remainder, so can be variable len
        batch_size = batch_in.size(0)
        preds = evaluator.predict(data_set, model, batch_in, batch_in_mask, batch_size)

        inp_strings = data_set.decode(batch_in, ignore_special=True)
        target_strings = data_set.decode(batch_target, ignore_special=True)
        pred_strings = data_set.decode(preds, ignore_special=True)
        for inp, targ, pred in zip(inp_strings, target_strings, pred_strings):
            print("".join(inp), "".join(targ), "".join(pred))


if __name__=='__main__':
    main()
