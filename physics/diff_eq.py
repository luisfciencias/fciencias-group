# template for numerical integration of differential equations
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))  # type: Figure, list[Axes]
for mu_param in mu_list:
    integrator_results = integrate.odeint(dynamical_system, y0, t, args=(mu_param,))
    df_res = pd.DataFrame(integrator_results, columns=['x', 'p'])
    axes[0].plot(t, df_res['x'], label=r"$\mu = {}$".format(mu_param))
    axes[1].plot(t, df_res['p'])
    axes[2].plot(df_res['x'], df_res['p'])
    plt.xlabel("$t$", fontsize=12)
    plt.ylabel("$X$", fontsize=12)

axes[0].legend(fontsize=12)
[ax.set_xlabel(x_label, fontsize=12) for ax, x_label in zip(axes, ["$t$", "$t$", "$x$"])]
[ax.set_ylabel(y_label, fontsize=12) for ax, y_label in zip(axes, ["$x$", "$p$", "$p$"])]

axes[0].set_title(r"$\ddot{x} - \mu (1-x^2) \dot{x} + x = 0$")

fig.tight_layout()
output_figure = "fig_output.png"
plt.savefig(output_figure)
print("Output to: {}".format(output_figure))
