from scipy.integrate import odeint

from helpers.plotting import *
from ode_functions.current import sodium_current
from ode_functions.diff_eq import ode_2d, ode_3d, ode_5d, voltage_clamp, default_parameters, current_voltage_curve, \
    pulse
from ode_functions.gating import *


def run():
    """
    Top level runner for figure 1
    :return: None
    """

    print("Running: Figure 1")

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

    __figure1_supplemental_files__()


def __figure1_supplemental_files__():
    """ Make supplemental figures using unmodified parameters taken directly from paper data"""

    paper_value = 5.92 / 1e6  # nS/cm^2 in mS/cm^2
    __figure1a__('Supplemental 1A(v1)', g_na=paper_value)
    plt.ylim([-1, 1])  # reset to make line visible
    save_fig("supp_1A_v1")

    __figure1a__('Supplemental 1A(v2)', g_na=5.92)  # assume units incorrect and meant mS/cm^2
    save_fig("supp_1A_v2")

    __figure1b__('Supplemental 1B(v1)', g_na=9.12, pulse_width=3)
    save_fig("supp_1B_v1")

    __figure1b__('Supplemental 1B(v2)', pulse_width=3)
    save_fig("supp_1B_v2")

    __figure1b__('Supplemental 1B(v3)', g_na=9.12)
    save_fig("supp_1B_v3")

    __figure1c__('Supplemental 1C', use_paper_shift=True)
    save_fig("supp_1C")

    __figure1d__('Supplemental 1D2', ix=1, use_paper_shift=True)  # ix=1 means use 5D model
    save_fig("supp_1D2")


def __figure1a__(title, g_na=5.92 * 0.514):
    """
    Compute IV curve for 2d and 3d ode model

    Based on the paper's reference of Seutin and Engel 2010 we believe what should have happened is Vm is clamped to
    -120 mV then Vm is clamped to a series of voltages: being returned to Vm=-120 each time. For each jump the peak Ina
    is computed

    Using the Paper's g_na = 5.92 we find a peak IV at -311 uA/cm^2 whereas the target is ~-160 uA/cm^2 so we rescale
    g_na to 5.92*160/311 which gives a peak IV curve at ~-160 uA/cm^2     todo: discuss in paper

    :param title: Plot title (panel label)
    :param g_na: Sodium conductance to use - since this figure requires modification optionally set different values
    :return: None
    """

    time_points = np.arange(0, 500)
    voltage = np.arange(-90, 60, 0.5)  # Compute IV curve for -100mV to 60mV every 0.5mV
    pre_v_clamp = -120
    ic = clamp_steady_state(pre_v_clamp)  # initial condition for the second clamp (after clamp to -120 mV)

    for ix, func in enumerate([ode_3d, ode_2d]):  # use 2d then 3d ode
        use_system_hs = True  # use the hs from the ode
        color = 'grey'
        linestyle = 'solid'

        if ix == 1:  # update parameters for 2d sstem
            ic = ic[:-1]  # drop hs dimension in 2d ode
            use_system_hs = False  # hs does not exist it will be set to 1
            color = 'black'
            linestyle = '--'

        current = current_voltage_curve(func, voltage, time_points, ic, use_system_hs=use_system_hs, g_na=g_na)
        plt.plot(voltage, current, color=color, linestyle=linestyle)

    make_legend(["3D", "2D"], loc='center left', bbox_to_anchor=(0.3, 1.05))
    set_properties(title, x_label="V (mV)", y_label="peak I$_{Na}$($\mu$A/cm$^2$)", x_tick=[-80, -40, 0, 40],
                   y_tick=[-160, 0])


def clamp_steady_state(v_clamp):
    """
    Determine the steady-state of ode_3d when clamped to a voltage

    :param v_clamp: Voltage to clamp to
    :return: Steady state of the neuron at v_clamp
    """
    state = odeint(voltage_clamp, [v_clamp, 1, 1], [0, 500], args=(default_parameters(), ode_3d))
    return state[-1, :]


def generate_clamp_times(positive_pulse_width=3):  # todo clean this up + probably refactor + docstring
    pulse_period = 100
    end_time = 500
    holding_potential = -70
    pulse_potential = 0

    pulse_high = False
    segment_start = [-100 + positive_pulse_width]
    segment_end = [0]
    pulse_value = [-70]

    while segment_end[-1] < end_time:
        if not pulse_high:
            segment_start.append(segment_end[-1])
            segment_end.append(segment_start[-1] + positive_pulse_width)
            pulse_value.append(pulse_potential)
            pulse_high = True
        else:
            segment_start.append(segment_end[-1])
            segment_end.append(segment_start[-1] + (pulse_period - positive_pulse_width))
            pulse_value.append(holding_potential)
            pulse_high = False

    pattern = {k: v for k, v in zip(segment_start, pulse_value)}
    return pattern, end_time


def __figure1b__(title, g_na=5.92, pulse_width=5):  # todo clean up
    """
    Periodic depolarizing pulses

    :param title: Plot title (panel label)
    :return: None
    """

    pattern, clamp_times = generate_clamp_times(positive_pulse_width=pulse_width)
    ic = clamp_steady_state(-80)

    state, time, waveform = pulse(voltage_clamp, 'v_clamp', pattern, 500, ic, clamp_function=ode_3d, g_na=g_na)
    v, h, hs = state.T  # unpack requires dimension as row (transpose)

    plt.plot(time, sodium_current(v, m_inf(v), default_parameters(g_na=g_na), h=h, hs=hs), 'k')

    set_properties(title, x_label="time (ms)", y_label="I$_{Na}$ (pA)", x_tick=[0, 200, 400], y_tick=[-200, 0],
                   x_limits=[-50, 450])


def __figure1c__(title, use_paper_shift=False):
    """
    Evaluation of f(h) or n

    :param title: Plot title (panel label)
    :return: None
    """

    parameters = default_parameters()

    t = np.arange(0, 4200, 0.1)
    ic = [-55, 0, 0, 0, 0]
    if not use_paper_shift:  # our replication version with modified tau_n
        state = odeint(ode_5d, ic, t, args=(parameters,), rtol=1e-3)  # need to set tighter tolerance for 5d system
    else:  # show version in paper with shift=40
        state = odeint(lambda x, t, p: ode_5d(x, t, p, shift=40), ic, t, args=(parameters,), rtol=1e-3)

    h = state[int(len(t) / 2):, 1]  # throw out first half to remove transient
    n = state[int(len(t) / 2):, 4]

    replicate_fit = np.poly1d(np.polyfit(h, n, 3))
    print(replicate_fit)

    plt.plot(h, n, c="grey")
    plt.plot(h, list(map(f, h)), "k")

    make_legend(["n", "n=f(h)"], loc='center left', bbox_to_anchor=(0.3, 1.05))
    set_properties(title, x_label="h", y_label="n", x_tick=[0, 0.2, 0.4, 0.6], y_tick=np.arange(0, 1, 0.2),
                   x_limits=[0, 0.7])


def __figure1d__(title, ix=0, use_paper_shift=False):
    """
    Show waveforms for 3d (ix=0) or 5d (ix=1) models

    :param title: Plot title (panel label)
    :param ix: Set the model to use 3d/5d (ix=0/1)
    :param use_paper_shift: Optional parameter to use paper shift in figure to highlight error
    :return: None
    """

    ode_function = [ode_3d, ode_5d][ix]  # different panels (ix) use a different ode function: set the appropriate one

    parameters = default_parameters()
    t = np.arange(0, 4300, 0.1)

    ic = [-55, 0, 0]
    if ix == 1:
        ic += [0, 0]  # if ix is 1 this appends an additions (0,0) to the inital conditions

    if not use_paper_shift:  # our replication version with modified tau_n
        state = odeint(ode_function, ic, t, args=(parameters,),
                       rtol=1e-3)  # need to set tighter tolerance for 5d system
    else:  # show version in paper with shift=40
        state = odeint(lambda x, t, p: ode_5d(x, t, p, shift=40), ic, t, args=(parameters,), rtol=1e-3)

    t_throw_away = np.where(t > 1000)[0][0]
    state = state[t_throw_away:, :]
    t = t[t_throw_away:] - t[t_throw_away]  # set t[0] to 0

    plt.plot(t, state[:, 0], "k")
    y_label = 'v (mV)'
    y_tick_label = None

    if ix > 0:
        y_label = ""
        y_tick_label = []

    set_properties(title, x_label="time (ms)", y_label=y_label, y_tick=[-80, -40, 0], y_limits=[-80, 20],
                   y_ticklabel=y_tick_label, x_tick=[0, 1000, 2000, 3000], x_limits=[0, 3000])
