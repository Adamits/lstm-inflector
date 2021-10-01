# LSTM Inflector

## Install

You can 'install' with the setup.py:

`python setup.py develop` will create a python module in your environment that updates as you update the code. It can then be imported like a regular python module:

```
import lstm_inflector
```

## Usage

Train a sequence to sequence LSTM for morphological inflection generation. An example is given in `experiments/run.sh`. See `train.py` for all options, including a bidirectional or unidirectional encoder, as well as including attention or not.

## Details

The Attention function is, I believe, from [Luong et al 2015](https://aclanthology.org/D15-1166/). We may want to implement Bahdanau attention as well/instead.
The input data format for the `sigmorphon_data` dataset setting is the same as the SIGMORPHON shared tasks: a file where each line is a tab-delimited string of (lemma, surface form, bundle of morphological tags).

## Requirements

This code assumes python 3.9 - or at least any version where the `click` library is in the standard python library.
