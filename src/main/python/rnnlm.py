import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import utils
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

class LanguageModelModule(nn.Module):
    def __init__(
        self,
        vocabulary_size=None,
        embedding_dim=None,
        dropout=None,
        rnn_dim=None,
        rnn_layers=None
        ):
        super(LanguageModelModule, self).__init__()

        self._embedding = nn.Embedding(
            vocabulary_size, embedding_dim
            )

        self._embedding_dropout = nn.Dropout(dropout)

        rnn_dropout = dropout / rnn_layers

        self._rnn = nn.LSTM(
            embedding_dim, rnn_dim, rnn_layers,
            bias=True, batch_first=True, dropout=rnn_dropout, bidirectional=True
            )

        cutoffs = [round(vocabulary_size/15), 3 * round(vocabulary_size/15)]
        self._softmax = nn.AdaptiveLogSoftmaxWithLoss(
            rnn_dim * 2,
            vocabulary_size,
            cutoffs=cutoffs,
            div_value=4
            )

    def forward(self, X, targets):      
        X = self._embedding(X)
        X = self._embedding_dropout(X)
        X, _ = self._rnn(X)
        X = X.reshape(-1, X.size()[2])
        targets = targets.reshape(targets.size()[0] * targets.size()[1])
        y = self._softmax(X, targets)
        return y

    def predict(self, X, targets):
        p = self.forward(X, targets).output
        p = p.reshape(-1, X.size()[1])
        return p

class LanguageModelTrainer(object):
    def __init__(self, config=None, vocabulary=None, gpu=None):
        self._config = config
        self._gpu = gpu
        self._device = "cuda:{}".format(self._gpu)
        self._model = LanguageModelModule(
            vocabulary_size=vocabulary.max_id() + 1,
            embedding_dim=self._config["EMBEDDING_DIM"],
            dropout=self._config["DROPOUT"],
            rnn_dim=self._config["RNN_DIM"],
            rnn_layers=self._config["RNN_LAYERS"]
            )
        self._model.to(self._device)
        self._model = DDP(self._model, device_ids=[self._gpu])
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=self._config["LEARNING_RATE"]
            )

    def train(
        self,
        train_dataset=None,
        eval_dataset=None,
        train_sampler=None,
        trace=None
        ):
        
        # Datasets and loading
        train_dataloader = utils.data.DataLoader(
            train_dataset,
            batch_size=self._config["BATCH_SIZE"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler
            )

        eval_dataloader = utils.data.DataLoader(
            eval_dataset,
            batch_size=self._config["BATCH_SIZE"],
            shuffle=True
            )

        # Training loop
        num_epochs = self._config["NUM_EPOCHS"]
        self._model.train()
        for epoch in range(num_epochs):
            loop = enumerate(train_dataloader)
            if trace:
                loop = tqdm(loop, total=len(train_dataloader), leave=True)

            # Training in epoch
            train_loss_accumulated = 0.0
            train_loss = None
            for i, (inputs, targets) in loop:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                self._optimizer.zero_grad()
                yhat = self._model(inputs, targets)
                loss = yhat.loss
                loss.backward()
                train_loss_accumulated += loss.item()
                self._optimizer.step()

                if trace:
                    loop.set_description(
                        "Epoch [{}/{}]".format(epoch + 1, num_epochs)
                        )
                if trace and (not i % 10):
                    train_loss = train_loss_accumulated / (i + 1)
                    loop.set_postfix(train_loss=train_loss)
        
            # Evaluating in epoch
            if trace:
                self._model.eval()
                eval_loss_accumulated = 0.0
                for inputs, targets in eval_dataloader:
                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)
                    yhat = self._model(inputs, targets)
                    eval_loss_accumulated += yhat.loss.item()
                eval_loss = eval_loss_accumulated / len(eval_dataloader)
                self._model.train()
                print(
                    "Training Loss: {:.4f}, Validation Loss: {:.4f}".format(
                        train_loss, eval_loss
                    )
                )

    def save(self, path):
        torch.save(self._model.module.state_dict(), path)

class Vocabulary(object):
    def __init__(self, path):
        self._word_counts = {}
        self._ids2words = {}
        self._words2ids = {}
        self.size = 1
        self.padding_id = 0
        self.unknown_id = 1
        self.unknown_word = "<unk>"
        self.padding_word = "<pad>"

        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "r") as fh:
                for line in fh.readlines():
                    line = line.strip()
                    words = line.split()
                    for word in words:
                        if word in [self.unknown_word, self.padding_word]:
                            continue
                        self._add(word)
        
        sorted_words = list(
            sorted(
                self._word_counts.keys(),
                key=lambda word: self._word_counts[word],
                reverse=True
                ))

        for word in sorted_words:
            self._register(word)

    def _add(self, word):
        if word in self._word_counts:
            self._word_counts[word] += 1
        else:
            self._word_counts[word] = 1

    def _register(self, word):
        self.size += 1
        self._ids2words[self.size] = word
        self._words2ids[word] = self.size

    def max_id(self):
        return self.size
    
    def id(self, word):
        if word in self._words2ids:
            return self._words2ids[word]
        else:
            return self.unknown_id

    def word(self, id):
        if id in self._ids2words:
            return self._ids2words[id]
        elif id == self.padding_id:
            return self.padding_word
        else:
            return self.unknown_word

class DataPreprocessor(object):
    def __init__(self, vocabulary=None, max_len=None):
        self.vocabulary = vocabulary
        self.max_len = max_len

    def process(self, sentence):
        words = sentence.split()
        inputs = words[:-1]
        input_indices = [self.vocabulary.padding_id] * self.max_len
        for i in range(min(self.max_len, len(inputs))):
            input_indices[i] = self.vocabulary.id(inputs[i])

        targets = words[1:]
        target_indices = [self.vocabulary.padding_id] * self.max_len
        for i in range(min(self.max_len, len(targets))):
            target_indices[i] = self.vocabulary.id(targets[i])

        return input_indices, target_indices

class LanguageModelDataset(utils.data.Dataset):
    def __init__(self, path, vocabulary=None, max_len=None):
        self._processor = DataPreprocessor(
            vocabulary=vocabulary,
            max_len=max_len
            )

        paths = []
        if os.path.isdir(path):
            for filename in os.listdir(path):
                paths.append(
                    os.path.join(path, filename)
                    )
        else:
            paths.append(path)

        X, y = [], []
        for file_path in paths:
            with open(file_path, "r") as fh:
                for line in fh.readlines():
                    line = line.strip()
                    context, target = self._processor.process(line)
                    X.append(context)
                    y.append(target)
        self._X = torch.LongTensor(X)
        self._y = torch.LongTensor(y)

    # number of rows in the dataset
    def __len__(self):
        return len(self._X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self._X[idx], self._y[idx]]

class LanguageModel(object):
    def __init__(
        self,
        model_path=None,
        config=None,
        vocabulary=None
        ):

        self._model = LanguageModelModule(
            vocabulary_size=vocabulary.max_id() + 1,
            embedding_dim=config["EMBEDDING_DIM"],
            dropout=config["DROPOUT"],
            rnn_dim=config["RNN_DIM"],
            rnn_layers=config["RNN_LAYERS"]
            )

        self._model.load_state_dict(
            torch.load(
                model_path, map_location=torch.device('cpu')
            )
        )

    def logprob_batch(self, dataset, batch_size=None):
        dataloader = utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
            )

        logprobs = []
        for inputs, targets in tqdm(dataloader):
            p = self._model.predict(inputs, targets).tolist()
            batch_logprobs = np.sum(p, axis=1)
            batch_logprobs = batch_logprobs.tolist()
            logprobs = logprobs + batch_logprobs

        return logprobs
    
    def prob_batch(self, dataset):
        return np.exp(self.logprob_batch(dataset))