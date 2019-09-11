import os

bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_len = 128
batch_size = 32
DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DIR, "twitter.csv")
MODEL_FILE = os.path.join(DIR, "model.h5")
