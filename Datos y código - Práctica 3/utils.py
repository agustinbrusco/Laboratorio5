from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import curve_fit


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
                      ) -> tuple[plt.figure, list[plt.axes]]:
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
    if tiempo.size % window_size == 0:
        t = np.reshape(tiempo,
                       (-1, window_size))
        ch = np.reshape(channel,
                        (-1, window_size))
    else:
        t = np.reshape(tiempo[:-(tiempo.size % window_size)],
                       (-1, window_size))
        ch = np.reshape(channel[:-(channel.size % window_size)],
                        (-1, window_size))
    min_ids = ch.argmin(axis=1, keepdims=True)
    t_min = np.take_along_axis(t, min_ids, axis=1).flatten()
    ch_min = np.take_along_axis(ch, min_ids, axis=1).flatten()
    max_ids = ch.argmax(axis=1, keepdims=True)
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
                      ) -> tuple[plt.figure, list[plt.axes]]:
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
    if channel.size % N == 0:
        ch = np.reshape(channel,
                        (-1, N))
    else:
        ch = np.reshape(channel[:-(channel.size % N)],
                        (-1, N))
    counts = np.sum(ch < umbral, axis=1)
    return counts


def plot_dist(t, v, T, peak_window, freq, umbral,
              ax=None, text_x=13.5, phantom=False):
    counts = count_in_period(t, v, T, freq, umbral, peak_window)
    unique_vals, count_vals = np.unique(counts, return_counts=True, )
    sum_count = count_vals.sum()
    avg_count = counts.mean()
    # print(mean_count)
    curve_vals = np.arange(unique_vals.min(), unique_vals.max()+1, 1)
    dist_med = count_vals/sum_count

    if ax is None:
        fig, ax = plt.subplots(1, 1,)
    if not phantom:
        dist_raw = poisson(avg_count).pmf(curve_vals)
        popt, pcov = curve_fit(poisson.pmf, unique_vals, dist_med, p0=avg_count)
        dist_fit = poisson(popt[0]).pmf(curve_vals)

        ax.stem(unique_vals, dist_med,
                linefmt="k", markerfmt="og", basefmt="None",
                label="Observada")
        ax.plot(curve_vals, dist_raw, '-r', lw=1.5, label="Paramétrica")
        ax.plot(curve_vals, dist_fit, '--C0', lw=1, label="Ajuste")
        ax.set_ylim(bottom=0)
        ax.text(text_x, ax.get_ylim()[-1]*0.9,
                rf"$T$ = {T*1e3:.2n} ms"+"\n" + rf"$\langle m \rangle$ = {avg_count:.3g}",
                va="top", ha="right", fontsize=12,
                bbox={'facecolor': 'white', 'boxstyle': 'round'})
    else:
        ax.scatter(unique_vals, dist_med, c="k", alpha=0.5, zorder=7, label="Rueda Fija")
    return ax
