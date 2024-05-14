import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import genfromtxt

def main():

    # Read data from csv
    data = genfromtxt('test.csv', delimiter=',')

    # Turn columns to row (easier to manipulate)
    data = np.transpose(data)

    # Get amount of channels
    channels = len(data) - 1

    # Set up figure
    fig = plt.figure()
    ax = fig.subplots(channels)

    # Seperate the time data from EEG channels
    x = data[0]
    y = data[1:channels+1]

    # Plot
    for i in range(channels):
        ax[i].plot(x, y[i], color = 'tab:red')
    plt.show()

if __name__ == '__main__':
    main()