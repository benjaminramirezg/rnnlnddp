import json
import pickle
import argparse
import pandas as pd
from rnnlm import LanguageModelModule, LanguageModel, LanguageModelDataset

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to JSON file with hyperparameters used to train the language model", type=str, required=True)
parser.add_argument("-m", "--model", help="Path to file with trained language model", type=str, required=True)
parser.add_argument("-v", "--vocabulary", help="Path to pickle file with vocabulary used to train language model", type=str, required=True)
parser.add_argument("-d", "--data", help="Path to file with input sentences", type=str, required=True)
parser.add_argument("-o", "--output", help="Path to output TSV file where results are stored", type=str, required=True)

args = parser.parse_args()

_DEFAULT_CONFIG = {
    "MAX_LEN": 50
    }

print("Reading config from {}...".format(args.config))
config = None
with open(args.config, "r") as fh:
    config = json.loads(fh.read())
    for default_key, default_value in _DEFAULT_CONFIG.items():
        if default_key not in config:
            config[default_key] = default_value

print("Loading vocabulary from {}...".format(args.vocabulary))
vocabulary = None
with open(args.vocabulary, 'rb') as handle:
    vocabulary = pickle.load(handle)

print("Loading language model from {}...".format(args.model))
lm = LanguageModel(
    model_path=args.model,
    vocabulary=vocabulary,
    config=config
    )

print("Creating dataset from {}...".format(args.data))
dataset = LanguageModelDataset(
    args.data, vocabulary=vocabulary, max_len=config["MAX_LEN"]
    )

print("Predicting...")
logprobs = lm.logprob_batch(
    dataset, batch_size=config["BATCH_SIZE"]
    )

print("Saving results in {}".format(args.output))
fh = open(args.data, "r")
texts = fh.readlines()
fh.close()

df = pd.DataFrame()
df["text"] = [t.strip() for t in texts]
df["logprob"] = logprobs
df.to_csv(args.output, sep="\t", index=False)