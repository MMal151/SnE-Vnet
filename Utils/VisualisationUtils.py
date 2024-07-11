import matplotlib.pyplot as plt
import pandas as pd

pd.options.plotting.backend = "plotly"

# Possible Options: "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
template = "seaborn"


def plot_column(df, column, type="hist", filename=None):
    if type == "bar":
        fig = df[column].plot.bar(template=template)
    elif type == "hist":
        fig = df[column].plot.hist(template=template)

    if filename is None:
        fig.show()
    else:
        fig.write_image(filename)


def show_history(history, validation: bool = False):
    if validation:
        # Loss
        fig, axes = plt.subplots(figsize=(20, 5))
        # Train
        axes.plot(history.epoch, history.history['loss'], color='r', label='Train')
        axes.plot(history.epoch, history.history['val_loss'], color='b', label='Val')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.legend()
        plt.savefig('loss.jpg')
        plt.show()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        # loss
        axes[0].plot(history.epoch, history.history['loss'])
        axes[0].set_title('Train')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
