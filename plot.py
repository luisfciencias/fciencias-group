import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.02, 10, 100)
y = np.sin(x)/x

plt.plot(x,y, 'g-')
plt.show()
