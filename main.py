# %%
# import numpy as np
import tensorflow as tf
from tensorflow import keras
import string, os 
import re
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pickle

AUTOTUNE = tf.data.experimental.AUTOTUNE


# %%
print(tf.__version__)


batch_size = 64
epochs = 200
latent_dim = 512
num_samples = 50000

# %%

df = pd.read_csv('topical_chat.csv')
print(df.head())

# %%
# Preprocessing

def process(text):

    text = text.lower().replace('?', '').replace('.', '').replace(',', '').replace('!', '').replace('\'', '').replace('\"', '').replace('(', '').replace(')', '').replace('-', '').replace(';', '').replace(':', '')
    text = "".join(v for v in text if v not in string.punctuation).strip()
    text = " ".join(text.split())

    return text


df .message.apply(process)

# %%
# Verctorization
inputs_text = []
targets_text = []
input_characters = set()
target_characters = set()

for conversation_index in tqdm(range(df.shape[0])):
    if conversation_index == 0:
        continue

    input_text = df.iloc(df.message[conversation_index - 1])
    target_text = df.iloc(df.message[conversation_index])

    if input_text.conversation_id == target_text.conversation_id:
        input_text = input_text.message
        target_text = target_text.message

        if len(input_text.split()) > 2 and \
            len(target_text.split()) > 0 \ and 