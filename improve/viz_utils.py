"""Plotting Functionality."""

from pathlib import Path
import matplotlib.pyplot as plt


def draw_loss(train_losses, test_losses, title, outpath: Path):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    fname = title + ".png"
    plt.savefig(outpath / fname)  # should be called before show method


def draw_pearson(pearsons, title, outpath: Path):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    fname = title + ".png"
    plt.savefig(outpath / fname)  # should be called before show method
