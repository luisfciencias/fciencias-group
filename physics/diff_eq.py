# template for numerical integration of differential equations
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

plt.rcParams['mathtext.fontset'] = 'cm'
import argparse
import sys
import time
import logging


def print_welcome(name):
    # TODO: Add your name to the author list
    logging.info("-----------------------------------------------")
    logging.info(" ___   ___    ___     __      __      _   ")
    logging.info("| __| / __|  |_  )   /  \    /  \    / |  ")
    logging.info("| _| | (__    / /   | () |  | () |   | |  ")
    logging.info("|_|   \___|  /___|   \__/    \__/    |_|  ")
    logging.info("")
    logging.info("---------------- FC1002 team -----------------")
    logging.info("")
    logging.info(" ==============================================")
    logging.info(" " + name)
    logging.info(" Authors: Luis Torres, Pablo Galaviz           ")
    logging.info(" ==============================================")
    logging.info("")


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
    dp_dt = mu * (1 - x ** 2) * p - x
    return dx_dt, dp_dt


if __name__ == "__main__":

    # store start time for benchmarking
    start_time = pd.to_datetime(time.time(), unit="s")

    # setup logger
    log_formatter = logging.Formatter('FC2001 %(levelname)s [%(asctime)s] | %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # setup arguments
    epilog_text = "FC2001 team "
    parser = argparse.ArgumentParser(description='Simple differential equation integrator', epilog=epilog_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--silent', action='store_true', help='Starts in silent mode, no message will be output.')
    parser.add_argument('-d', '--debug', action='store_true', help='Shows debug info')
    parser.add_argument('-o', '--output_path', type=str, help='Input path to output file', default="result.png",
                        required=True)
    parser.add_argument('-x', '--initial_position', type=float, help='Initial position', default=0, required=False)
    parser.add_argument('-p', '--initial_momentum', type=float, help='Initial momentum', default=1, required=False)
    parser.add_argument('-t', '--integration_time', type=float, help='Integration time', default=2 * np.pi,
                        required=False)
    parser.add_argument('-k', '--delta_time', type=float, help='Step size', default=0.01, required=False)

    # parse arguments and set logger
    args = parser.parse_args()
    consoleHandler = logging.StreamHandler(sys.stdout)

    if args.debug:
        root_logger.setLevel(logging.DEBUG)

    if args.silent:
        consoleHandler.setLevel(logging.ERROR)

    consoleHandler.setFormatter(log_formatter)
    root_logger.addHandler(consoleHandler)

    print_welcome("diff_eq")

    # parameters
    mu_list = [0.0, 1.0, 2.0]

    logging.debug("mu parameters: %f, %f, %f.  ", mu_list[0], mu_list[1], mu_list[2])

    # initial conditions
    x0, p0 = args.initial_position, args.initial_momentum
    y0 = x0, p0

    # time grid
    t = np.arange(0, args.integration_time, args.delta_time)

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
    output_figure = args.output_path
    plt.savefig(output_figure)
    logging.info("Output to: {}".format(output_figure))
    logging.info("Total computation time: %s", str(pd.to_datetime(time.time(), unit="s") - start_time))
