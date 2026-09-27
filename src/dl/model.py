"""
Branch B — DL Temporal Model: CNN -> BiLSTM -> Self-Attention -> Dense.

Architecture choice (documented, as required):
  CNN (1D conv)   : cheap local-pattern extraction over the flow-sequence
                    (e.g. a short burst of failed logins).
  BiLSTM          : captures longer-range forward/backward temporal
                    dependencies across the window (slow multi-stage
                    attacks span many flows).
  Self-attention  : a lightweight additive/Bahdanau-style attention layer
                    over the BiLSTM outputs, rather than a full Transformer
                    encoder. A full multi-head Transformer encoder was
                    considered (as sketched in the design doc) but a single
                    additive-attention layer over BiLSTM states was chosen
                    here because it is computationally realistic on CPU-only
                    hardware, keeps the parameter count small enough to
                    train in minutes rather than hours on the small window
                    sizes used, and still exposes genuine, inspectable
                    per-time-step attention weights for Stage 4's
                    explainability layer (a full Transformer's multi-head
                    attention would need to be reduced/averaged to a single
                    weight-per-step anyway for that purpose). Swapping in a
                    `keras.layers.MultiHeadAttention` block is a drop-in
                    change (see `build_transformer_variant` below) if more
                    compute is available.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="tri_hybrid_xai_ids")
class AdditiveAttention(layers.Layer):
    """Bahdanau-style additive attention over a sequence of BiLSTM states.
    Produces a context vector AND the per-time-step attention weights
    (needed for Stage 4 attention visualization / cross-explainer check)."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)
        self.v = layers.Dense(1)

    def call(self, hidden_states):
        # hidden_states: (batch, time, features)
        score = self.v(tf.nn.tanh(self.W(hidden_states)))          # (batch, time, 1)
        weights = tf.nn.softmax(score, axis=1)                      # (batch, time, 1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)    # (batch, features)
        return context, tf.squeeze(weights, axis=-1)                # weights: (batch, time)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


def build_model(window_size: int, n_features: int, n_classes: int, cfg: dict) -> keras.Model:
    dc = cfg["dl_branch"]
    inputs = keras.Input(shape=(window_size, n_features), name="flow_sequence")

    x = layers.Conv1D(dc["cnn_filters"], dc["cnn_kernel_size"], padding="same", activation="relu")(inputs)
    x = layers.Dropout(dc["dropout"])(x)

    x = layers.Bidirectional(layers.LSTM(dc["lstm_units"], return_sequences=True))(x)
    x = layers.Dropout(dc["dropout"])(x)

    context, attn_weights = AdditiveAttention(dc["attention_units"], name="attention")(x)

    d = layers.Dense(64, activation="relu")(context)
    d = layers.Dropout(dc["dropout"])(d)
    outputs = layers.Dense(n_classes, activation="softmax", name="prediction")(d)

    model = keras.Model(inputs=inputs, outputs=[outputs, attn_weights], name="cnn_bilstm_attention")
    # `attention` is an inspectable side-output (for Stage 4 XAI), not a training
    # target, so it gets no loss (loss=None) and is skipped by loss_weights=0.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=dc["learning_rate"]),
        loss=["sparse_categorical_crossentropy", None],
        loss_weights=[1.0, 0.0],
        metrics={"prediction": "accuracy"},
    )
    return model


def build_transformer_variant(window_size: int, n_features: int, n_classes: int, cfg: dict) -> keras.Model:
    """CNN -> BiLSTM -> single-head Transformer-style self-attention encoder block.
    More expensive; kept as an opt-in alternative (see model.py docstring)."""
    dc = cfg["dl_branch"]
    inputs = keras.Input(shape=(window_size, n_features))
    x = layers.Conv1D(dc["cnn_filters"], dc["cnn_kernel_size"], padding="same", activation="relu")(inputs)
    x = layers.Bidirectional(layers.LSTM(dc["lstm_units"], return_sequences=True))(x)
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=dc["attention_units"])(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    pooled = layers.GlobalAveragePooling1D()(x)
    d = layers.Dense(64, activation="relu")(pooled)
    outputs = layers.Dense(n_classes, activation="softmax")(d)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=dc["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def get_attention_weights(model: keras.Model, X_seq: np.ndarray) -> np.ndarray:
    """Returns (n_samples, window_size) attention weights for a batch of sequences."""
    _, attn = model.predict(X_seq, verbose=0)
    return attn
