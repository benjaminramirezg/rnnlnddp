import os
import json
import pickle
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from rnnlm import LanguageModelTrainer, Vocabulary, LanguageModelDataset

_TRAINING_DATA_TMP_PATH = "./traindata.tmp.pkl"
_EVALUATION_DATA_TMP_PATH = "./evaldata.tmp.pkl"

_DEFAULT_CONFIG = {
    "BATCH_SIZE": 2000,
    "NUM_EPOCHS": 2,
    "EMBEDDING_DIM": 300,
    "DROPOUT": 0.5,
    "RNN_DIM": 250,
    "RNN_LAYERS": 3,
    "MAX_LEN": 50,
    "LEARNING_RATE": 0.004
    }

def get_config(path):
    config = None
    with open(path, "r") as fh:
        config = json.loads(fh.read())
        for default_key, default_value in _DEFAULT_CONFIG.items():
            if default_key not in config:
                config[default_key] = default_value
    return config

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    trace_flag = True if rank == 0 else False

    dist.init_process_group(
        backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )

    if trace_flag:
        print("Reading config from {}...".format(args.config))
    config = get_config(args.config)

    if trace_flag:
        print("Loading vocabulary from {}...".format(args.vocabulary))
    vocabulary = None
    with open(args.vocabulary, 'rb') as handle:
        vocabulary = pickle.load(handle)

    if trace_flag:
        print("Loading training dataset from {}...".format(
            _TRAINING_DATA_TMP_PATH
            ))
    train_dataset = None
    with open(_TRAINING_DATA_TMP_PATH, 'rb') as handle:
        train_dataset = pickle.load(handle)

    if trace_flag:
        print("Loading evaluation dataset from {}...".format(
            _EVALUATION_DATA_TMP_PATH
            ))
    eval_dataset = None
    with open(_EVALUATION_DATA_TMP_PATH, 'rb') as handle:
        eval_dataset = pickle.load(handle)
    
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
        )

    if trace_flag:
        print("Creating model with config: {}...".format(config))
    lm_trainer = LanguageModelTrainer(
        config=config,
        vocabulary=vocabulary,
        gpu=gpu
    )
    
    if trace_flag:
        print("Training...")
    lm_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_sampler=train_sampler,
        trace=trace_flag
        )

    if trace_flag:
        print("Saving model in {}...".format(args.model))
        lm_trainer.save(
            args.model
            )

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument("-t", "--traindata", help="Path to folder with data to train the language model", type=str, required=True)
    parser.add_argument("-e", "--evaldata", help="Path to folder with data to eval the language model", type=str, required=True)
    parser.add_argument("-c", "--config", help="Path to JSON file with hyperparameters to train the language model", type=str, required=True)
    parser.add_argument("-m", "--model", help="Path to output file where language model will be saved", type=str, required=True)
    parser.add_argument("-p", "--vocabulary", help="Path to output pickle file where vocabulary will be saved", type=str, required=True)
    parser.add_argument('-n', '--nodes', default=1, type=int, help='number of nodes')
    parser.add_argument('-g', '--gpus', default=4, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('-ma', '--ma', default="localhost", type=str, help='Master address')
    parser.add_argument('-mp', '--mp', default="12355", type=str, help='Master port')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.ma
    os.environ['MASTER_PORT'] = args.mp

    print("Reading config from {}...".format(args.config))
    config = get_config(args.config)

    print("Saving vocabulary in {}...".format(args.vocabulary))
    vocabulary = None
    with open(args.vocabulary, 'wb') as handle:
        vocabulary = Vocabulary(path=args.traindata)
        pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Loading training dataset from {}...".format(args.traindata))
    with open(_TRAINING_DATA_TMP_PATH, 'wb') as handle:
        train_dataset = LanguageModelDataset(
            args.traindata, vocabulary=vocabulary, max_len=config["MAX_LEN"]
            )
        pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Loading eval dataset from {}...".format(args.evaldata))
    with open(_EVALUATION_DATA_TMP_PATH, 'wb') as handle:
        eval_dataset = LanguageModelDataset(
            args.evaldata, vocabulary=vocabulary, max_len=config["MAX_LEN"]
            )
        pickle.dump(eval_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Running training in {} parallel processes...".format(
        args.nodes * args.gpus
        ))
    mp.spawn(train, nprocs=args.gpus, args=(args,))

    os.remove(_EVALUATION_DATA_TMP_PATH)
    os.remove(_TRAINING_DATA_TMP_PATH)

if __name__ == '__main__':
    main()