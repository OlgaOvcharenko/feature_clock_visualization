import pandas
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.datasets import fetch_openml


if __name__ == '__main__':
    import random
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(int)

    data = mnist.data
    data["class"] = mnist.target

    parallel_coordinates(data, 'class')
    plt.savefig("plots/parallel_axis.png")
