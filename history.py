import json
from matplotlib import pyplot as plt
import tensorflow.keras as keras


def plot_history(history, metrics=None):
    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics is None:
        metrics = [x for x in history.history.keys() if x[:4] != 'val_']
    if len(metrics) == 0:
        print('No metrics to display.')
        return

    x = history.epoch

    rows = 1
    cols = len(metrics)
    count = 0

    plt.figure(figsize=(12 * cols, 8))
    for metric in sorted(metrics):
        count += 1
        plt.subplot(rows, cols, count)
        plt.plot(x, history.history[metric], label='Train')
        val_metric = f'val_{metric}'
        if val_metric in history.history.keys():
            plt.plot(x, history.history[val_metric], label='Validation')
        plt.title(metric.capitalize())
        plt.legend()
    plt.show()


def add_history(old_hist, new_hist):
    old_hist.epoch.extend(new_hist.epoch)
    old_hist.params = new_hist.params
    for k in old_hist.history.keys():
        old_hist.history[k].extend(new_hist.history[k])


def save_history(history, model_name):
    hist_out = {}
    hist_out['epoch'] = history.epoch
    hist_out['history'] = history.history
    hist_out['params'] = history.params
    with open(f'{model_name}.history', 'w') as outfile:
        json.dump(hist_out, outfile)


def load_history(model_name, model_format=''):
    with open(f'{model_name}.history', 'r') as f:
        hist = json.load(f)
    history = keras.callbacks.History()
    history.epoch = hist['epoch']
    history.history = hist['history']
    history.params = hist['params']
    model = keras.models.load_model(f'{model_name}{model_format}')
    history.set_model(model)

    return history