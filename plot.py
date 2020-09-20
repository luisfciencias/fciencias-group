#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot():
    x = np.linspace(0.02, 10, 100)
    y = np.sin(x)/x

    plt.plot(x,y, 'g-')
    plt.show()

if __name__ == '__main__':
    exit(plot())
