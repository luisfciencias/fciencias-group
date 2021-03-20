# template for numerical integration of differential equations
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'


def dynamical_system(y, ti, mu):
    """
    Van der Pol oscillator dynamical system
    :param y: derivatives vector
    :param ti: time array / grid for numerical integration
    :param mu: damping term in Van der Pol equation. mu = 0 recovers classical oscillator
    :return: tuple of derivatives
    """
    x, p = y
    dx_dt = p
    dp_dt = mu * (1 - x**2) * p - x
    return dx_dt, dp_dt


# parameters
mu_list = [0.0, 1.0, 2.0]

# initial conditions
x0, p0 = 0, 1
y0 = x0, p0

# time grid
t = np.linspace(0, 2 * np.pi, 500)

# integrator for different parameters
for mu_param in mu_list:
    ret = integrate.odeint(dynamical_system, y0, t, args=(mu_param,))
    df_res = pd.DataFrame(ret, columns=['x', 'p'])
    plt.plot(t, df_res['x'], label=r"$\mu = {}$".format(mu_param))
    plt.xlabel("$t$", fontsize=12)
    plt.ylabel("$X$", fontsize=12)
plt.legend(fontsize=12)
plt.show()
