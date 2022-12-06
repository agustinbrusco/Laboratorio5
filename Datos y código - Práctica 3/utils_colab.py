from typing import Union, Tuple, List
import numpy as np
import matplotlib.pyplot as plt


def plot_projected_dist(x: np.ndarray, y: np.ndarray,
                        dx: np.ndarray, dy: np.ndarray,
                        bins: np.ndarray, density: bool = False,
                        fmt: str = ".", fig: Union[plt.figure, None] = None):
    if fig is None:
        fig, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)
        fig.set_size_inches((8, 3.5))
        axs[0].set_xlabel("Tiempo [ms]")
        axs[0].set_ylabel("Tensión [mV]")
    else:
        axs = fig.axes
    axs[0].errorbar(x=x, xerr=dx,
                    y=y, yerr=dy,
                    fmt=fmt, ms=6,
                    ecolor="k", capsize=2,
                    alpha=0.5, zorder=10)
    axs[1].hist(x=y, bins=bins, orientation="horizontal", density=density,
                histtype="step", lw=1.5, alpha=0.7, zorder=10)
    return fig, axs


def plot_measurements(tiempo: np.ndarray,
                      channel: np.ndarray,
                      min_val: float,
                      max_val: float,
                      med_index: int = 0,
                      n_datos: int = 2000,
                      fig: Union[plt.figure, None] = None,
                      density: bool = False,
                      ) -> Tuple[plt.figure, List[plt.axes]]:
    """"""
    t = tiempo[med_index*n_datos:(med_index + 1)*n_datos]
    ch = channel[med_index*n_datos:(med_index + 1)*n_datos]
    # Define resolution
    temp_res = 1/250e3  # s
    volt_res = 1.05*(max_val - min_val)/(2**16)  # V
    # arrays for plotting
    x = (t - t[0])*1e3  # ms
    dx = temp_res*1e3/2  # ms
    y = ch*1e3  # mV
    dy = volt_res*1e3/2  # mV
    bin_range = np.arange(min_val*1e3, max_val*1e3, 2)  # mV
    bins = bin_range[(y.max() + 2 >= bin_range) & (y.min() - 2 <= bin_range)]
    if n_datos < 200:
        fig, axs = plot_projected_dist(x, y, dx, dy, bins, density, ".-", fig)
    else:
        fig, axs = plot_projected_dist(x, y, dx, dy, bins, density, ".", fig)
    return fig, axs


def get_peaks(tiempo, channel, window_size):
    t = np.reshape(tiempo, (-1, window_size))
    ch = np.reshape(channel, (-1, window_size))
    min_ids = ch.argmin(axis=1)
    t_min = np.take_along_axis(t, min_ids, axis=1).flatten()
    ch_min = np.take_along_axis(ch, min_ids, axis=1).flatten()
    max_ids = ch.argmax(axis=1)
    t_max = np.take_along_axis(t, max_ids, axis=1).flatten()
    ch_max = np.take_along_axis(ch, max_ids, axis=1).flatten()
    time = np.concatenate([t_min, t_max])
    chan = np.concatenate([ch_min, ch_max])
    # order = np.argsort(time)
    return time, chan


def plot_custom_peaks(tiempo: np.ndarray,
                      channel: np.ndarray,
                      min_val: float,
                      max_val: float,
                      med_index: int = 0,
                      n_datos: int = 2000,
                      fig: Union[plt.figure, None] = None,
                      density: bool = False,
                      window_size: int = 10
                      ) -> Tuple[plt.figure, List[plt.axes]]:
    """"""
    t = tiempo[med_index*n_datos:(med_index + 1)*n_datos]
    ch = channel[med_index*n_datos:(med_index + 1)*n_datos]
    t_p, ch_p = get_peaks(t, ch, window_size)
    # Define resolution
    temp_res = 1/250e3  # s
    volt_res = 1.05*(max_val - min_val)/(2**16)  # V
    # Arrays for plotting
    x = (t_p - t[0])*1e3  # ms
    dx = temp_res*1e3/2  # ms
    y = ch_p*1e3  # mV
    dy = volt_res*1e3/2  # mV
    bin_range = np.arange(min_val*1e3, max_val*1e3, 2)  # mV
    bins = bin_range[(y.max() + 2 >= bin_range) & (y.min() - 2 <= bin_range)]
    fig, axs = plot_projected_dist(x, y, dx, dy, bins, density, "v", fig)
    return fig, axs


def count_in_period(tiempo,  # s
                    channel,  # V
                    T,  # s
                    f,  # Hz
                    umbral,  # V
                    window_size,
                    ):
    N = int(T*f/window_size)
    # print(N)
    # t = np.reshape(tiempo, (-1, N))
    ch = np.reshape(channel, (-1, N))
    counts = np.sum(ch < umbral, axis=1)
    return counts
