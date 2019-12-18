import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm, trange


import torch
from torch import cuda
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from seqeval.metrics import f1_score



# load data and fill na
data = pd.read_csv("/workspace/data/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

# get unique words from the "Word" column
words = list(set(data["Word"].values))
n_words = len(words)

# get unique tags fromn the "Tag" column
tags_vals = list(set(data["Tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}

print(cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(n_gpu)