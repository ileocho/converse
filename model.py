"https://www.kaggle.com/code/saurabhprajapat/conversational-chatbot-using-encoder-and-decoder"

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import activations, layers, models, preprocessing
from keras.utils import np_utils, pad_sequences

print(tf.version)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_path = 'data.csv'

inputs = []
targets = []

# Opening  and reading the file

with open(data_path) as f:
    data_t = f.read().split('\n')

for line in data_t[: min(600, len(data_t)-1)]:

    input_text = line.split('\t')[0]
    target = line.split('\t')[1]

    inputs.append(input_text)
    targets.append(target)

# print('type of input text ', type(input_text))
# print('type of target text ', type(target))

# Tokenizer

zip_l = list(zip(inputs, targets))
lines = pd.DataFrame(zip_l, columns=["input", "output"])

input_lines = []
for line in lines.input:
    input_lines.append(line)

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(input_lines)
tokenized_input_text = tokenizer.texts_to_sequences(input_lines)

lenght_list = list()
for token_seq in  tokenized_input_text:
    lenght_list.append(len(token_seq))
max_input_lenght = np.array(lenght_list).max()
# print(f'max output lenght is {max_input_lenght}')

# Encoder

padded_input_lines = pad_sequences(tokenized_input_text, maxlen=max_input_lenght, padding='post')
encoder_input_data = np.array(padded_input_lines)

input_word_dict = tokenizer.word_index
num_input_tokens = len(input_word_dict) + 1
# print(f'Number of input tokens : {num_input_tokens}')

# print(encoder_input_data)

# Decoder

output_lines = list()
for line in lines.output:
    output_lines.append("<START>" + line + "<END>")

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(output_lines)
tokenized_output_text = tokenizer.texts_to_sequences(output_lines)

lenght_list = list()
for token_seq in  tokenized_output_text:
    lenght_list.append(len(token_seq))
max_output_length = np.array(lenght_list).max()
# print(f'max output lenght is {max_output_length}')


padded_output_lines = pad_sequences(tokenized_output_text, maxlen=max_output_length, padding='post')
encoder_output_data = np.array(padded_output_lines)
decoder_input_data = np.array(padded_output_lines)

output_word_dict = tokenizer.word_index
num_output_tokens = len(output_word_dict) + 1
# print(f'Number of output tokens : {num_output_tokens}')

# print(encoder_output_data)

decoder_target_data = list()
for token_seq in tokenized_output_text:
    decoder_target_data.append(token_seq[1:])

padded_output_lines = pad_sequences(decoder_target_data, maxlen=max_output_length, padding='post')
onehot_output_lines = np_utils.to_categorical(padded_output_lines, num_output_tokens)
decoder_target_data = np.array(onehot_output_lines)
# print(f'Decoder output shape -> {decoder_target_data.shape}')

#Model

encoder_inputs = tf.keras.layers.Input(shape=(None, ))
encoder_embeddings = tf.keras.layers.Embedding(num_input_tokens, 256, mask_zero=True)(encoder_inputs)
encoder_output, state_h, state_c = tf.keras.layers.LSTM(256, return_state=True, recurrent_dropout=0.2, dropout=0.2)(encoder_embeddings)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None, ))
decoder_embedding = tf.keras.layers.Embedding(num_output_tokens, 256, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(256, return_state=True, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(num_output_tokens, activation=tf.nn.softmax)
output = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy')

model.summary()

model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=124, epochs=500)
model.save('model.h5')

model = tf.keras.models.load_model('model.h5', custom_objects={"Model": tf.keras.models.Model})

def make_inference_model():
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    decoder_state_input_h = tf.keras.layers.Input(shape=(256, ))
    decoder_state_input_c = tf.keras.layers.Input(shape=(256, ))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_state_inputs,
                                          [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

def str_to_tokens(sentence):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(input_word_dict[word])
    return pad_sequences([tokens_list], maxlen=max_input_lenght, padding='post')

enc_model, dec_model = make_inference_model()
for epoch in range(encoder_input_data.shape[0]):
    states_values = enc_model.predict(str_to_tokens(input('User: ')))
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = output_word_dict['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in output_word_dict.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
            stop_condition = True

        empty_target_seq = np.zeros( ( 1 , 1 ) )
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ]

    print( "Bot: " +decoded_translation.replace(' end', '') )
    print()
