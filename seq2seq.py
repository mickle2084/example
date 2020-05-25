from __future__ import print_function
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import tensorflow as tf
import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

config = tf.ConfigProto(device_count = {'GPU':1, 'CPU':10})
sess=tf.Session(config=config)
keras.backend.set_session(sess)
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Embedding, Bidirectional, TimeDistributed
import matplotlib.pyplot as plt
import joblib
import re

params = joblib.load('params.txt')

class Dataloader:
    def __init__(self, f, batch_size, steps_per_epoch, params):
        self.f = f
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.params = params
        self.batch_y = [str(i) for i in range(batch_size)]

    def __call__(self):
        with open(self.f, 'r', encoding='utf-8') as f:
            count = 0
            while True:
                count += 1
                if count >= self.steps_per_epoch + 1:
                    f.seek(0)
                    count = 0
                for batch in range(self.batch_size):
                    self.batch_y[batch] = f.readline()

                batch_x = self.clean_punctuation(self.batch_y)
                yield self.convert_data(batch_x, self.batch_y, self.params)

    @staticmethod
    def clean_punctuation(text):
        reg = re.compile('[^а-яёА-Я0-9\- ]')
        clean_text = [reg.sub('', i) for i in text]
        new_text = []
        for line in clean_text:
          line = line.replace('.', '')
          line = line.replace('!', '')
          line = line.replace('?', '')
          line = line.replace('  ', ' ')
          line = line.replace('   ', ' ')
        return clean_text

    @staticmethod
    def change_chars(text):
        changed_text = []
        for line in text:
            if line[2] == ' ':
                line = line.replace(line[3], 'й', 2)
            else:
                line = line.replace(line[2], 'й', 2)
            changed_text.append(line)
        return changed_text


    @staticmethod
    def convert_data(input_batch, target_batch, params):
        target_texts = []
        input_texts = []
        for line in target_batch:
            target_text = line
            target_text = '\t' + target_text + '\n'
            target_texts.append(target_text)
        for line1 in input_batch:
            input_text = line1.replace('\n', '')
            input_texts.append(input_text)
        encoder_input_data = np.zeros(
            (len(input_texts), params['max_encoder_seq_length'], params['num_encoder_tokens']),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), params['max_decoder_seq_length'], params['num_decoder_tokens']),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), params['max_decoder_seq_length'], params['num_decoder_tokens']),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, params['input_token_index'][char]] = 1.
            encoder_input_data[i, t + 1:, params['input_token_index'][' ']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, params['target_token_index'][char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, params['target_token_index'][char]] = 1.
            decoder_input_data[i, t + 1:, params['target_token_index'][' ']] = 1.
            decoder_target_data[i, t:, params['target_token_index'][' ']] = 1.
        x = [encoder_input_data, decoder_input_data]
        y = decoder_target_data

        return x, y

from keras.callbacks import *
batch_size = 150
checkpoint = ModelCheckpoint('callbacks/saved-model-{epoch: 02d}.hdf5', monitor='val_acc', verbose=1, mode='max')

tensorboard_callback = TensorBoard(log_dir='logs/', update_freq=100, batch_size=batch_size,
                                                       histogram_freq=0, write_graph=True)


data = "clean.txt"
val_data = "val_data.txt"
from subprocess import run, PIPE
#steps_per_epoch_train = int(run(['wc', '-l', 'data_1000.txt?dl=0'], stdout=PIPE).stdout.split()[0]) // batch_size
steps_per_epoch_train = 1299929//batch_size
#int(run(["wc", "-l", data], stdout=PIPE).stdout.split()[0]) // batch_size
val_steps = 9
#int(run(["wc", "-l", val_data], stdout=PIPE).stdout.split()[0]) // 10

#train_loader = Dataloader('data_1000.txt?dl=0', batch_size, steps_per_epoch_train, params)()
train_loader = Dataloader(data, batch_size, steps_per_epoch_train, params)()
val_loader = Dataloader(val_data, 10, val_steps, params)()

epochs = 300
latent_dim = 256
encoder_inputs = Input(shape=(None, params['num_encoder_tokens']))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, params['num_decoder_tokens']))

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(params['num_decoder_tokens'], activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator=train_loader,
                    epochs = epochs,
                    steps_per_epoch=steps_per_epoch_train,
                    validation_data = val_loader,
                    validation_steps = val_steps,
                    max_queue_size = 5,
                    callbacks=[checkpoint, tensorboard_callback])


