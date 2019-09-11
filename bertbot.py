import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from tokenization import FullTokenizer
from tensorflow.keras import backend as K
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from constants import bert_path, max_seq_len, batch_size
from constants import bert_path, max_seq_len, batch_size, DATA_FILE, MODEL_FILE


def bert_tokenizer(sess):
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file = sess.run(tokenization_info["vocab_file"])
    do_lower_case = sess.run(tokenization_info["do_lower_case"])
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


sess = tf.Session()
tokenizer = bert_tokenizer(sess)


def process_sample(tokenizer, text, max_seq_len):
    tokenized = tokenizer.tokenize(text)
    if len(tokenized) > max_seq_len - 2:
        tokenized = tokenized[0 : max_seq_len - 2]
    tokens = []
    tokens.append("[CLS]")
    tokens.extend(tokenized)
    tokens.append("[SEP]")
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(token_ids)
    seg_ids = [0] * len(token_ids)
    while len(token_ids) < max_seq_len:
        token_ids.append(0)
        input_mask.append(0)
        seg_ids.append(0)
    return np.asarray(token_ids), np.asarray(input_mask), np.asarray(seg_ids)


class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10
            )
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


def build_model(num_classes, max_seq_len=max_seq_len):
    in_id = tf.keras.layers.Input(shape=(max_seq_len,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_len,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_len,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs)
    dropout = tf.keras.layers.Dropout(0.5)(bert_output)
    dense = tf.keras.layers.Dense(128, activation="relu")(dropout)
    output = tf.keras.layers.Dense(num_classes, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=output)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=["accuracy"],
    )
    model.summary()
    return model


def make_dataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    def _process_string(X):
        X = X.numpy().decode()
        tok_ids, in_mask, seg_ids = process_sample(tokenizer, X, max_seq_len)
        return tok_ids, in_mask, seg_ids

    def _process(X, y):
        tok_ids, in_mask, seg_ids = tf.py_function(
            _process_string, [X], (tf.uint32, tf.uint32, tf.uint32)
        )
        tok_ids.set_shape(max_seq_len)
        in_mask.set_shape(max_seq_len)
        seg_ids.set_shape(max_seq_len)
        y.set_shape([])
        return (tok_ids, in_mask, seg_ids), y

    dataset = dataset.map(_process)
    dataset = dataset.shuffle(buffer_size=2000).batch(batch_size).prefetch(batch_size)
    dataset = dataset.repeat()
    return dataset


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


if __name__ == "__main__":
    print("loading data")
    # import the dataset
    with open(DATA_FILE, mode="r", encoding="ascii", errors="ignore") as csvfile:
        intents = pd.read_csv(csvfile, header=None)
        X = np.asarray(list(intents[5]))
        y = list(intents[0])

    # encoding labels
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    print("partitationing data")
    # partitationing dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print("creating dataset objects")
    # creating tf.data.Dataset objects
    train = make_dataset(X_train, y_train)
    val = make_dataset(X_val, y_val)
    test = make_dataset(X_test, y_test)

    print("building model")
    model = build_model(le.classes_.shape[0])

    print("initializing session")
    initialize_vars(sess)

    print("training model")
    stopping_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_FILE, monitor="val_loss", save_best_only=True, mode="min"
    )
    model.fit(
        train,
        validation_data=val,
        callbacks=[stopping_early, checkpoint],
        validation_steps=X_val.shape[0] / batch_size,
        steps_per_epoch=X_train.shape[0] / (2*batch_size),
        epochs=10,
    )

    eval_results = model.evaluate(test, steps=X_test.shape[0] / batch_size)
    print("bertbot's accuracy on the test set is " + str(eval_results[1]))
