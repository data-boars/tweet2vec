import pickle as pkl
from collections import OrderedDict

import numpy as np

from .settings_char import MAX_LENGTH


class BatchTweets:
    def __init__(
        self,
        data,
        targets,
        labeldict,
        batch_size=128,
        max_classes=1000,
        test=False,
    ):
        # convert targets to indices
        if targets is not None:
            if not test:
                tags = []
                for l in targets:
                    tags.append(
                        labeldict[l]
                        if l in labeldict and labeldict[l] < max_classes
                        else 0
                    )
            else:
                tags = []
                for line in targets:
                    tags.append(
                        [
                            labeldict[l]
                            if l in labeldict and labeldict[l] < max_classes
                            else 0
                            for l in line
                        ]
                    )
            self.targets = tags
        else:
            self.targets = None

        self.batch_size = batch_size
        self.data = data

        self.prepare()
        self.reset()

    def prepare(self):
        self.indices = np.arange(len(self.data))
        self.curr_indices = np.random.permutation(self.indices)

    def reset(self):
        self.curr_indices = np.random.permutation(self.indices)
        self.curr_pos = 0
        self.curr_remaining = len(self.curr_indices)

    def __next__(self):
        if self.curr_pos >= len(self.indices):
            self.reset()
            raise StopIteration()

        # current batch size
        curr_batch_size = np.minimum(self.batch_size, self.curr_remaining)

        # indices for current batch
        curr_indices = self.curr_indices[
            self.curr_pos : self.curr_pos + curr_batch_size
        ]
        self.curr_pos += curr_batch_size
        self.curr_remaining -= curr_batch_size

        # data and targets for current batch
        x = [self.data[ii] for ii in curr_indices]
        if self.targets is None:
            return x, [-1] * len(x)

        y = [self.targets[ii] for ii in curr_indices]
        return x, y

    def __iter__(self):
        return self


def prepare_data(seqs_x, chardict, n_chars=1000):
    """
    Prepare the data for training - add masks and remove infrequent characters
    """
    seqsX = []
    for cc in seqs_x:
        seqsX.append(
            [
                chardict[c] if c in chardict and chardict[c] <= n_chars else 0
                for c in list(cc)
            ]
        )
    seqs_x = seqsX

    lengths_x = [len(s) for s in seqs_x]

    n_samples = len(seqs_x)

    x = np.zeros((n_samples, MAX_LENGTH)).astype("int32")
    x_mask = np.zeros((n_samples, MAX_LENGTH)).astype("float32")
    for idx, s_x in enumerate(seqs_x):
        if len(s_x) > MAX_LENGTH:
            s_x = s_x[:MAX_LENGTH]
        x[idx, : lengths_x[idx]] = s_x
        x_mask[idx, : lengths_x[idx]] = 1.0

    return np.expand_dims(x, axis=2), x_mask


def build_dictionary(text):
    """
    Build a character dictionary
    text: list of tweets
    """
    charcount = OrderedDict()
    for cc in text:
        chars = list(cc)
        for c in chars:
            if c not in charcount:
                charcount[c] = 0
            charcount[c] += 1
    chars = list(charcount.keys())
    freqs = list(charcount.values())
    sorted_idx = np.argsort(freqs)[::-1]

    chardict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        chardict[chars[sidx]] = idx + 1

    return chardict, charcount


def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, "wb") as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)


def build_label_dictionary(targets):
    """
    Build a label dictionary
    targets: list of labels, each item may have multiple labels
    """
    labelcount = OrderedDict()
    for l in targets:
        if l not in labelcount:
            labelcount[l] = 0
        labelcount[l] += 1
    labels = list(labelcount.keys())
    freqs = list(labelcount.values())
    sorted_idx = np.argsort(freqs)[::-1]

    labeldict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        labeldict[labels[sidx]] = idx + 1

    return labeldict, labelcount
