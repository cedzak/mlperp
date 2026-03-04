#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 00:17:20 2025; @author: sylwia
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class TimeAttention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(1,),  # Zmiana na skalarną wartość, broadcastowalną do dowolnego SEQLEN
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )

    def call(self, inputs):
        # inputs: [BATCHSIZE, SEQLEN, features]
        scores = tf.matmul(inputs, self.W) + self.b  # Broadcasting self.b do [BATCHSIZE, SEQLEN, 1]
        e = tf.keras.activations.tanh(scores)
        alpha = tf.nn.softmax(e, axis=1)
        context = inputs * alpha
        return context if self.return_sequences else tf.reduce_sum(context, axis=1)

    def get_ta_konfig(self):
        return {**super().get_ta_konfig(), "return_sequences": self.return_sequences}




@tf.keras.utils.register_keras_serializable()
class DirectionalTimeAttention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            trainable=True
        )
        self.W_grad = self.add_weight(
            name="W_grad",
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            trainable=True
        )
        self.b = self.add_weight(
            name="b",
            shape=(1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )

    def call(self, x):
        gradients = x[:, 1:, :] - x[:, :-1, :] # ja: = [0, 0, wektor dlugosci liczby cech]
        gradients = tf.pad(gradients, [[0, 0], [1, 0], [0, 0]], "CONSTANT")
        value_scores = tf.matmul(x, self.W)
        grad_scores = tf.matmul(gradients, self.W_grad)
        combined_scores = tf.nn.tanh(value_scores + grad_scores + self.b)
        attention_weights = tf.nn.softmax(combined_scores, axis=1)
        weighted_output = x * attention_weights
        return weighted_output if self.return_sequences else tf.reduce_sum(weighted_output, axis=1)

    def get_ta_konfig(self):
        return {**super().get_ta_konfig(), "return_sequences": self.return_sequences}




@tf.keras.utils.register_keras_serializable()
class LongDirectionalTimeAttention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            trainable=True
        )
        self.W_grad_short = self.add_weight(
            name="W_grad_short",
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            trainable=True
        )
        self.W_grad_long = self.add_weight(
            name="W_grad_long",
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            trainable=True
        )
        self.b = self.add_weight(
            name="b",
            shape=(1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )

    def call(self, x):
        timesteps = tf.shape(x)[1]
        long_lag = tf.minimum(5, timesteps - 1)

        gradients_short = x[:, 1:, :] - x[:, :-1, :]
        gradients_short = tf.pad(gradients_short, [[0, 0], [1, 0], [0, 0]], "CONSTANT")
        
        # gradients: Tensor do dopełnienia (kształt [BATCHSIZE, SEQLEN - 1, features]).
           # [[0, 0], [1, 0], [0, 0]]: Określa, ile zer dodać wzdłuż każdego wymiaru:
           #  [0, 0] dla osi batch (nic nie dodajemy).
           #  [1, 0] dla osi time (dodajemy 1 zero na początku, nic na końcu).
           #  [0, 0] dla osi features (nic nie dodajemy).
           #  "CONSTANT": Wartości dopełnienia to zera (domyślnie).

        start_indices = tf.stack([0, long_lag, 0])
        sizes = tf.stack([-1, timesteps - long_lag, -1])
        x_long_lag_end = tf.slice(x, start_indices, sizes)

        start_indices = tf.stack([0, 0, 0])
        sizes = tf.stack([-1, timesteps - long_lag, -1])
        x_start_long_lag = tf.slice(x, start_indices, sizes)

        gradients_long = x_long_lag_end - x_start_long_lag
        gradients_long = tf.pad(gradients_long, [[0, 0], [long_lag, 0], [0, 0]], "CONSTANT")

        value_scores = tf.matmul(x, self.W)
        grad_scores_short = tf.matmul(gradients_short, self.W_grad_short)
        grad_scores_long = tf.matmul(gradients_long, self.W_grad_long)
        combined_scores = tf.nn.tanh(value_scores + grad_scores_short + grad_scores_long + self.b)
        attention_weights = tf.nn.softmax(combined_scores, axis=1)
        weighted_output = x * attention_weights
        return weighted_output if self.return_sequences else tf.reduce_sum(weighted_output, axis=1)

    def get_ta_konfig(self):
        return {**super().get_ta_konfig(), "return_sequences": self.return_sequences}




