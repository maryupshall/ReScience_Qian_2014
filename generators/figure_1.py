from scipy.integrate import odeint

from helpers.plotting import *
from ode_functions.current import sodium_current
from ode_functions.diff_eq import ode_2d, ode_3d, ode_5d, voltage_clamp, default_parameters
from ode_functions.gating import *


def run():
    """
    Top level runner for figure 1
    :return: None
    """

    init_figure(size=(7, 3))
    plt.subplot2grid((2, 6), (0, 0), colspan=2, rowspan=1)
    __figure1a__('A')

    plt.subplot2grid((2, 6), (0, 2), colspan=2, rowspan=1)
    __figure1b__('B')

    plt.subplot2grid((2, 6), (0, 4), colspan=2, rowspan=1)
    __figure1c__('C')

    for ix, col in enumerate([0, 3]):
        plt.subplot2grid((2, 6), (1, col), colspan=3, rowspan=1)
        __figure1d__('D' + str(ix + 1), ix=ix)

    save_fig("1")


def __figure1a__(title):
    """
    Compute IV curve for 2d and 3d ode model

    :param title: Plot title (panel label)
    :return: None
    """

    parameters = default_parameters(
        g_na=5.92 / 1.5)  # need to divide given value by 1.5 to get correct graph todo include in paper
    # number does not match and units correct here not paper

    ic = [-70, 0.815]  # start sim at -70, 0.815
    time = np.arange(0, 100, 0.1)  # run sim for 100 ms save every 100us
    voltage = np.arange(-100, 60, 0.5)  # Compute IV curve for -100mV to 60mV every 0.5mV

    ode_functions = [ode_2d, ode_3d]  # use 2d then 3d ode

    for ix, func in enumerate(ode_functions):
        current = np.zeros(voltage.shape)
        for iy, v in enumerate(voltage):
            state = odeint(voltage_clamp, ic + ix * [1], time, args=(parameters, func))
            if ix == 0:
                hs = 1
            else:
                hs = state[:, 2]

            current[iy] = np.min((sodium_current(v, m_inf(v), parameters, h=state[:, 1], hs=hs)))

        color = 'black' if ix == 0 else 'grey'
        linestyle = '--' if ix == 0 else 'solid'
        plt.plot(voltage, current, color=color, linestyle=linestyle, zorder=-ix)
    make_legend(["2D", "3D"], loc='center left', bbox_to_anchor=(0.3, 1.05))

    set_properties(title, x_label="V (mV)", y_label="peak I$_{Na}$($\mu$A/cm$^2$)", x_tick=[-80, -40, 0, 40],
                   y_tick=[-160, 0])


def __figure1b__(title):
    """
    Periodic depolarizing pulses

    :param title: Plot title (panel label)
    :return: None
    """

    clamp_current = np.array([])
    all_time = np.array([])

    # packet = [[-100, 0],
    #           [0, 3],
    #           [3, 100],
    #           [100, 103],
    #           [103, 200],
    #           [200, 203],
    #           [203, 300],
    #           [300, 303],
    #           [303, 400],
    #           [400, 403],
    #           [403, 500],
    #           [500, 503],
    #           [503, 600]]

    last_set = 500
    # todo make more apparent
    start_times = np.ravel(list(zip(np.arange(0, last_set + 1, 100), np.arange(3, last_set + 4, 100))))
    end_times = np.ravel(list(zip(np.arange(3, last_set + 100 + 1, 100), np.arange(100, last_set + 100 + 4, 100))))
    packet = list(zip(start_times, end_times))
    packet.insert(0, (-100, 0))

    parameters = default_parameters(g_na=9.12 / 1.3)  # need to divide given value get correct graph
    # todo backwards, number does not match and units correct here not paper

    ic = [-70, 0, 0]
    for (t0, t1) in packet:
        time = np.arange(t0, t1, 0.05)  # evaluate simulation between t0 and t1
        all_time = np.concatenate((all_time, time))  # keep track of contiguous simulation

        state = odeint(voltage_clamp, ic, time, args=(parameters, ode_3d))
        v = state[:, 0]
        h = state[:, 1]
        hs = state[:, 2]

        clamp_current = np.concatenate((clamp_current, sodium_current(v, m_inf(v), parameters, h=h, hs=hs)))
        ic = [0 if ic[0] == -70 else -70, h[-1], hs[-1]]  # move clamp to 0 if currently clamped to -70 (vise versa)

    plt.plot(all_time, clamp_current, 'k')
    set_properties(title, x_label="time (ms)", y_label="I$_{Na}$ (pA)", x_tick=[0, 200, 400], y_tick=[-200, 0],
                   x_limits=[-50, 600])


def __figure1c__(title):
    """
    Evaluation of f(h) or n

    :param title: Plot title (panel label)
    :return: None
    """

    parameters = default_parameters()

    t = np.arange(0, 4200, 0.1)
    ic = [-55, 0, 0, 0, 0]
    state = odeint(ode_5d, ic, t, args=(parameters,), rtol=1e-6, atol=1e-6)

    h = state[int(len(t) / 2):, 1]  # throw out first half to remove transient
    n = state[int(len(t) / 2):, 4]

    replicate_fit = np.poly1d(np.polyfit(h, n, 3))
    print(replicate_fit)

    plt.plot(h, n, c="grey")
    plt.plot(h, list(map(f, h)), "k")
    set_properties(title, x_label="h", y_label="n", x_tick=[0, 0.2, 0.4, 0.6], y_tick=np.arange(0, 1, 0.2),
                   x_limits=[0, 0.7])
    make_legend(["n", "n=f(h)"], loc='center left', bbox_to_anchor=(0.3, 1.05))


def __figure1d__(title, ix=0):
    """
    Show waveforms for 3d (ix=0) or 5d (ix=1) models

    :param title: Plot title (panel label)
    :param ix: Set the model to use 3d/5d (ix=0/1)
    :return: None
    """

    ode_functions = [ode_3d, ode_5d]
    parameters = default_parameters()
    t = np.arange(0, 4300, 0.1)

    ode_function = ode_functions[ix]

    ic = [-55, 0, 0] + ix * [0, 0]  # if ix is 1 this appends an additions (0,0) to the inital conditions
    state = odeint(ode_function, ic, t, args=(parameters,), atol=1e-3)

    state = state[13000:, :]
    t = t[13000:]
    t -= t[0]  # set initial time to 0
    plt.plot(t, state[:, 0], "k")
    y_label = 'v (mV)'
    y_tick_label = None

    if ix > 0:
        y_label = ""
        y_tick_label = []

    set_properties(title, x_label="time (ms)", y_label=y_label, y_tick=[-80, -40, 0], y_limits=[-80, 20],
                   y_ticklabel=y_tick_label, x_tick=[0, 1000, 2000, 3000], x_limits=[0, 3000])
