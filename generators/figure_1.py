from functools import partial

from ode_functions.diff_eq import *
from ode_functions.gating import *
from plotting import *


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
        __figure1d__('D' + str(ix + 1), panel=ix)

    save_fig("1")

    __figure1_supplemental_files__()


def __figure1_supplemental_files__():
    """ Make supplemental figures using unmodified parameters taken directly from paper data"""

    """
    If we take the paper's value of 5.92 nS and a 1cm^2 patch is equivalent to 5.92/1e6 mS/cm^2
    This gives a plot scaled to almost 0 because this is a negligible effective channel density
    """
    paper_value = 5.92 / 1e6  # nS/cm^2 in mS/cm^2
    __figure1a__('Supplemental 1A(v1)', g_na=paper_value)
    plt.ylim([-1, 1])  # reset to make line visible
    save_fig("supp_1A_v1")

    """
    If we assume the paper mis-reported the number 5.92mS/cm^2 as 5.92nS/cm^2 then we get a plot scaled by ~2
    The "onset of current" at ~-50 mV and the reversal potential at ~60mV is preserved
    """
    __figure1a__('Supplemental 1A(v2)', g_na=5.92)  # assume units incorrect and meant mS/cm^2
    save_fig("supp_1A_v2")

    """
    Similarly to figure 1A they report 9.12nS/cm^2 again we will assume this 9.12nS but 9.12mS/cm^2 additionally if we
    use a pule width of 3ms as reported we see the current is far too large
    """
    __figure1b__('Supplemental 1B(v1)', g_na=9.12, pulse_width=3)
    save_fig("supp_1B_v1")

    """
    If we adjust g_na and leave the pulse width at 3ms we get an appropriate current scale but the current does not 
    decay enough.
    """
    __figure1b__('Supplemental 1B(v2)', pulse_width=3)
    save_fig("supp_1B_v2")

    """
    If we leave g_na at 9.12mS/cm^2 and set the pulse width to 5ms we still see a current that's ~2x too large
    """
    __figure1b__('Supplemental 1B(v3)', g_na=9.12)
    save_fig("supp_1B_v3")

    """
    If we use the original equation for tau_n (with the shift being 40mV) then we see a very distorted limit cycle in 
    the n,h phase space
    """
    __figure1c__('Supplemental 1C', use_modified_tau_n=True)
    save_fig("supp_1C")

    """
    If we use the original equation for tau_n (with the shift being 40mV) then we see a very distorted voltage trace
    Incidentally this looks more like the GABA neuron in their cited Seutin and Engel paper (2010) with a rounded trough
    see figure 7B
    """
    __figure1d__('Supplemental 1D2', panel=1, use_modified_tau_n=True)  # ix=1 means use 5D model
    save_fig("supp_1D2")


def __figure1a__(title, g_na=5.92 * 0.514, v_reset=-120):
    """
    Compute IV curve for 2d and 3d ode model

    Based on the paper's reference of Seutin and Engel 2010 we believe what should have happened is Vm is clamped to
    -120 mV then Vm is clamped to a series of voltages: being returned to Vm=-120 each time. For each jump the peak Ina
    is computed

    Using the Paper's g_na = 5.92 we find a peak IV at -311 uA/cm^2 whereas the target is ~-160 uA/cm^2 so we rescale
    g_na to 5.92*160/311 which gives a peak IV curve at ~-160 uA/cm^2     todo: discuss in paper

    :param title: Plot title (panel label)
    :param g_na: Optional sodium conductance to use: defaults to working parameter
    :param v_reset: Optional reset potential: defaults to working parameter
    :return: None
    """

    """Compute simulation for 500ms and try 1mV increments between -90mV and 60mV"""
    time_points = np.arange(500)
    voltage = np.arange(-90, 60)

    """Compute an initial condition that is the model clamped at v_reset"""
    holding_condition = clamp_steady_state(v_reset)

    """Perform the same voltage clamp experiment """
    for model in [ode_3d, ode_2d]:  # use 2d then 3d ode

        """Set different parameters of the function is the 2d ode"""
        if model is ode_2d:
            holding_condition = holding_condition[:-1]  # drop hs dimension in 2d

        """Set plot properties for the 2 models"""
        color, linestyle = ('grey', 'solid') if model is ode_3d else ('black', '--')

        current = current_voltage_curve(model, voltage, time_points, holding_condition, g_na=g_na)
        plt.plot(voltage, current, color=color, linestyle=linestyle)

    make_legend(["3D", "2D"], loc='center left', bbox_to_anchor=(0.3, 1.05))
    set_properties(title, x_label="V (mV)", y_label="peak I$_{Na}$($\mu$A/cm$^2$)", x_tick=[-80, -40, 0, 40],
                   y_tick=[-160, 0])


def generate_clamp_pattern_1b(end_time, pulse_width=5):
    """
    Helper function to generate the waveform for figure 1b

    Figure 1b requires a clamp waveform with timing requirements that are annoying to generate this function
    encapsulate that operation

    :param end_time: End of the simulation
    :param pulse_width: Optional width to the pulse (ms): defaults to 5ms
    :return: Returns the clamp pattern in the "pattern" format
    """

    pulse_period = 100
    clamp_potential = {'low': -70, 'high': 0}

    clamped_high = False
    segment_start = [-100 + pulse_width]
    segment_end = [0]
    clamp_potential_sequence = [clamp_potential['low']]

    """Generate start and end times for the waveform until the simulation end is hit"""
    while segment_end[-1] < end_time:
        """If the membrane is currently clamped low: compute pulse time and set new clamp parameters"""
        segment_start.append(segment_end[-1])
        if not clamped_high:
            remaining_time = pulse_width
            clamp_potential_sequence.append(clamp_potential['high'])
            clamped_high = True
        else:
            remaining_time = pulse_period - pulse_width
            clamp_potential_sequence.append(clamp_potential['low'])
            clamped_high = False
        segment_end.append(segment_start[-1] + remaining_time)  # end: remaining time after new start time

    """Return a dictionary of start_time:clamp_potential key-value pairs"""
    return {k: v for k, v in zip(segment_start, clamp_potential_sequence)}


def __figure1b__(title, g_na=5.92, pulse_width=5):  # todo clean up
    """
    Compute the perodic step current response from figure 1B

    Clamp to membrane potential to -80 then depolarize and rest the membrane potential to 0mV and -70mV every 100ms with
    5mV pulses to 0mV.

    In the original paper g_na is 9.12 mS/cm^2 and pulses are 5ms to reproduce their results we use g_na = 5.92 and a
    5 ms pulse

    :param title: Plot title (panel label)
    :param g_na:  Optional sodium conductance to use: defaults to working parameter
    :param pulse_width:  Optional pulse width (ms) to use: defaults to working parameter
    :return: None
    """

    end_time = 500
    pre_pulse_potential = -80
    pattern = generate_clamp_pattern_1b(end_time, pulse_width=pulse_width)
    pre_pulse_holding_condition = clamp_steady_state(pre_pulse_potential)

    solution, time, waveform = pulse(voltage_clamp, 'v_clamp', pattern, end_time, pre_pulse_holding_condition,
                                     clamp_function=ode_3d, g_na=g_na)  # todo: still needs refactoring

    i_na = sodium_current(solution, default_parameters(g_na=g_na))
    plt.plot(time, i_na, 'k')
    set_properties(title, x_label="time (ms)", y_label="I$_{Na}$ (pA)", x_tick=[0, 200, 400], y_tick=[-250, -200, 0],
                   x_limits=[-50, 450])


def __figure1c__(title, use_modified_tau_n=True):
    """
    Compute limit cycle in n,h phase space for the 5d model and compute the approximation n=f(h) for 1C

    :param title: Plot title (panel label)
    :param use_modified_tau_n: Optional parameter to use the original tau_n which does not work. Defaults to our tau_n
    :return: None
    """

    """Run the simulation for 4200ms and start at an arbitrary point "close" to the limit cycle"""
    time_points = np.arange(0, 4200, 0.1)
    initial_condition = [-55, 0, 0, 0, 0]  # Does not need to lie on limit cycle since we throw away transient
    parameters = default_parameters()

    """Solve 5d model with our corrected tau_n"""
    if use_modified_tau_n:  # our replication version with modified tau_n
        state = odeint(ode_5d, initial_condition, time_points, args=(parameters,), rtol=1e-3)
    else:
        original_ode_5d = partial(ode_5d, shift=40)  # todo put shift to be bool
        state = odeint(original_ode_5d, initial_condition, time_points, args=(parameters,), rtol=1e-3)

    """Extract h and n and discard the first half due to transient"""
    h = state[int(len(time_points) / 2):, 1]
    n = state[int(len(time_points) / 2):, 4]

    """Perform a fit on n = f(h)"""
    replicate_fit = np.poly1d(np.polyfit(h, n, deg=3))
    print(replicate_fit)

    plt.plot(h, n, c="grey")
    plt.plot(h, f(h), "k")  # todo rename f

    make_legend(["n", "n=f(h)"], loc='center left', bbox_to_anchor=(0.3, 1.05))
    set_properties(title, x_label="h", y_label="n", x_tick=[0, 0.2, 0.4, 0.6], y_tick=np.arange(0, 1, 0.2),
                   x_limits=[0, 0.7])


def __figure1d__(title, panel=0, use_modified_tau_n=True):
    """
    Show waveforms for 3d (ix=0) or 5d (ix=1) models

    :param title: Plot title (panel label)
    :param panel: Set the model to use 3d/5d (panel=0/1)
    :param use_modified_tau_n: Optional parameter to use the original tau_n which does not work. Defaults to our tau_n
    :return: None
    """

    """Select appropriate model depending on the panel"""
    model = [ode_3d, ode_5d][panel]

    """Run the simulation for 4200ms and start at an arbitrary point "close" to the limit cycle"""
    parameters = default_parameters()
    time_points = np.arange(0, 4200, 0.1)
    initial_condition = [-55, 0, 0]

    """If we're using a 5d model then add two dimensions the the initial condition"""
    if model == ode_5d:
        initial_condition += [0, 0]

    """Solve 5d model with our corrected tau_n"""
    if use_modified_tau_n:  # our replication version with modified tau_n
        state = odeint(model, initial_condition, time_points, args=(parameters,), rtol=1e-3)
    else:
        original_ode_5d = partial(ode_5d, shift=40)  # todo put shift to be bool
        state = odeint(original_ode_5d, initial_condition, time_points, args=(parameters,), rtol=1e-3)

    """Throw away the first 1000ms of the simulation"""
    t_throw_away = np.where(time_points > 1000)[0][0]
    state = state[t_throw_away:, :]
    time_points = time_points[t_throw_away:] - time_points[t_throw_away]  # set t[0] to 0

    plt.plot(time_points, state[:, 0], "k")
    y_label = 'V (mV)' if panel == 0 else ""
    y_tick_label = None if panel == 0 else []

    set_properties(title, x_label="time (ms)", y_label=y_label, y_tick=[-80, -40, 0], y_limits=[-80, 20],
                   y_ticklabel=y_tick_label, x_tick=[0, 1000, 2000, 3000], x_limits=[0, 3000])
