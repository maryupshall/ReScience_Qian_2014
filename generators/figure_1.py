from scipy.integrate import odeint

from helpers.plotting import *
from ode_functions.current import sodium_current
from ode_functions.diff_eq import ode_2d, ode_3d, ode_5d, voltage_clamp, default_parameters
from ode_functions.gating import *


def run():
    init_figure(size=(6, 3))
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
    parameters = default_parameters(g_na=0.00000592 / 1.5)  # need to divide given value by 1.5 to get correct graph
    parameters.append(None)
    time = np.arange(0, 100, 0.1)
    voltage = np.arange(-100, 60, 0.5)

    ode_functions = [ode_2d, ode_3d]

    current = np.zeros((len(voltage), 2))
    for ix, func in enumerate(ode_functions):
        parameters[-1] = func
        for iy, v in enumerate(voltage):
            ic = [-70, 0.815]
            state = odeint(voltage_clamp, ic + ix * [1], time, args=(parameters,))
            if ix == 0:
                hs = 1
            else:
                hs = state[:, 2]

            current[iy, ix] = 1e6 * np.min((sodium_current(v, m_inf(v), parameters, h=state[:, 1], hs=hs)))

    plt.plot(voltage, current[:, 1], color='grey')
    plt.plot(voltage, current[:, 0], 'k--')

    set_properties(title, x_label="V (mV)", y_label="peak I$_{Na}$", x_tick=[-80, -40, 0, 40], y_tick=[-160, 0])


def __figure1b__(title):
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
    start_times = np.ravel(list(zip(np.arange(0, last_set + 1, 100), np.arange(3, last_set + 4, 100))))
    end_times = np.ravel(list(zip(np.arange(3, last_set + 100 + 1, 100), np.arange(100, last_set + 100 + 4, 100))))
    packet = list(zip(start_times, end_times))
    packet.insert(0, (-100, 0))

    parameters = default_parameters(g_na=0.00000592)  # need to divide given value by 2 to get correct graph
    parameters.append(ode_3d)
    ic = [-70, 0, 0]

    for (t0, t1) in packet:
        time = np.arange(t0, t1, 0.05)  # off
        all_time = np.concatenate((all_time, time))

        state = odeint(voltage_clamp, ic, time, args=(parameters,))
        v = state[:, 0]
        h = state[:, 1]
        hs = state[:, 2]

        clamp_current = np.concatenate((clamp_current, sodium_current(v, m_inf(v), parameters, h=h, hs=hs)))
        ic = [0 if ic[0] == -70 else -70, h[-1], hs[-1]]  # move clamp to 0 if currently clamped to -70 (vise versa)

    plt.plot(all_time, 1e6 * clamp_current, 'k')
    set_properties(title, x_label="time (ms)", y_label="I$_{Na}$ (pA)", x_tick=[0, 200, 400], y_tick=[-200, 0],
                   x_limits=[-50, 600])


def __figure1c__(title):
    parameters = default_parameters()

    t = np.arange(0, 4200, 0.01)
    ic = [-55, 0, 0, 0, 0]
    state = odeint(ode_5d, ic, t, args=(parameters,), rtol=1e-6, atol=1e-6)

    h = state[200000:, 1]
    n = state[200000:, 4]

    plt.plot(h, n, c="grey")
    plt.plot(h, list(map(f, h)), "k")
    set_properties(title, x_label="h", y_label="n", x_tick=[0, 0.2, 0.4, 0.6], y_tick=np.arange(0, 1, 0.2), x_limits=[0, 0.7])
    make_legend(["n", "n=f(h)"])


def __figure1d__(title, ix=0):
    ode_functions = [ode_3d, ode_5d]
    parameters = default_parameters()
    t = np.arange(0, 4300, 0.01)

    ode_function = ode_functions[ix]

    ic = [-55, 0, 0] + ix * [0, 0]
    state = odeint(ode_function, ic, t, args=(parameters,), atol=1e-3)

    plt.plot(t, state[:, 0], "k")
    y_label = 'v (mV)'
    y_tick_label = None

    if ix > 0:
        y_label = ""
        y_tick_label = []
    set_properties(title, x_label="time (ms)", y_label=y_label, y_tick=[-80, -40, 0], y_limits=[-80, 20],
                   y_ticklabel=y_tick_label)
