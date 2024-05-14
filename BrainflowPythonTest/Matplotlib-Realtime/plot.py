import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import genfromtxt

def main():

    data = genfromtxt('test.csv', delimiter=',')
    data = np.transpose(data)
    channels = len(data) - 1

    fig = plt.figure()
    ax = fig.subplots(channels)

    x = data[0]
    y = data[1:channels+1]

    for i in range(channels):
        ax[i].plot(x, y[i], color = 'tab:red')
    plt.show()

if __name__ == '__main__':
    main()