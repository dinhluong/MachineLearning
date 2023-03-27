import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import json
from history import plot_history, save_history

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def retrieve_data():
    precipitation_url = 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110-pcp-all-12-1900-2020.json'
    cache_dir = './'
    cache_subdir = "data"
    filename = "monthly_precipitation.json"
    precip_file = tf.keras.utils.get_file(filename, precipitation_url, cache_dir=cache_dir, cache_subdir=cache_subdir)

    with open(precip_file) as f:
        precip_raw = json.load(f)

    precip_data = precip_raw['data']

    time = []
    precipitation = []

    for month in precip_data:
        time.append(month)
        precipitation.append(float(precip_data[month]['value']))

    return np.array(precipitation), time

def split_data(sequence, time, split_time):
    main_seq = sequence[:split_time]
    main_time = time[:split_time]

    extra_seq = sequence[split_time:]
    extra_time = sequence[split_time:]

    print(f'Splitting into {len(main_seq)} main examples and {len(extra_seq)} extra samples. ')

    return main_seq, main_time, extra_seq, extra_time

def wrangle_data(sequence, data_split, examples, batch_size):
    examples = examples + 1
    seq_expand = tf.expand_dims(sequence, -1)
    dataset = tf.data.Dataset.from_tensor_slices(seq_expand)
    dataset = dataset.window(examples, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda b: b.batch(examples))
    dataset = dataset.map(lambda x: (x[:-1],x[-1]))

    if data_split == 'train':
        dataset = dataset.shuffle(10000)
    else:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def compile_model(new_model):
    new_model.compile(optimizer='adam', loss='mae', metrics=[tf.metrics.RootMeanSquaredError()])
    print(new_model.summary())
    return new_model

def rnn_model():
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu'),
        tf.keras.layers.LSTM(48),
        tf.keras.layers.Dense(36, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)

def plot_sequence(time, sequences, start=0, end=None):
    y_min, y_max = (0.0, 1.0)
    if len(np.shape(sequences)) == 1:
        sequences = [sequences]
    time = time[start:end]
    plt.figure(figsize=(28, 8))
    for sequence in sequences:
        y_max = max(np.max(sequence), y_max)
        y_min = min(np.min(sequence), y_min)
        sequence = sequence[start:end]
        plt.plot(time, sequence)
    plt.ylim(y_min, y_max)
    plt.xlim(np.min(time), np.max(time))

def save_model(model, name, history, test_data):
    test_loss, test_rmse = model.evaluate(test_data)

    # Save model information
    save_name = f'models/precip/{name}-{len(history.epoch):02d}-{test_rmse:0.4f}'
    model.save(f'{save_name}.h5')

    # Save history information
    save_history(history, save_name)

def show_predictions(trained_model, predict_sequence, true_values, predict_time, begin=0, end=None):
    predictions = trained_model.predict(predict_sequence)
    predictions = predictions[:, -1].reshape(len(predictions))
    plot_sequence(predict_time, (true_values, predictions), begin, end)
    return predictions


if __name__ == '__main__':
    monthly_precip, time_dates = retrieve_data()
    time_steps = list(range(len(time_dates)))
    min_precip = np.min(monthly_precip)
    max_precip = np.max(monthly_precip)
    precip_norm = (monthly_precip - min_precip)/(max_precip - min_precip)

    test_split = time_dates.index('200001')
    valid_split = time_dates.index('199001')


    train_valid_sp, train_valid_time, test_sp, test_time = split_data(precip_norm, time_steps, test_split)
    train_sp, train_time, valid_sp, valid_time = split_data(train_valid_sp, train_valid_time, valid_split)

    examples = 6
    batch_size = 16

    train_data = wrangle_data(train_sp, 'train', examples, batch_size)
    valid_data = wrangle_data(valid_sp, 'valid', examples, batch_size)
    test_data = wrangle_data(test_sp, 'test', examples, batch_size)

    model_name = 'rnn'
    early_stop = EarlyStopping('val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=f'ckpts/precip/{model_name}/' + '{epoch:02d}-{val_loss:.4f}')

    model = rnn_model()
    history = model.fit(train_data, epochs=100, validation_data=valid_data,
                        callbacks=[early_stop, checkpoint])

    save_model(model, model_name, history, test_data)


    show_predictions(model, test_data, test_sp[examples:], test_time[examples:])
    plot_history(history)