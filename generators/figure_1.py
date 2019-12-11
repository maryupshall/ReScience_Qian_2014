from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from ode_functions.current import sodium_current
from ode_functions.diff_eq import (
    steady_state_when_clamped,
    ode_3d,
    ode_2d,
    resize_initial_condition,
    current_voltage_curve,
    pulse,
    default_parameters,
    ode_5d,
    solve_ode,
    f_approx,
)
from plotting import init_figure, save_fig, set_properties


def run():
    """Top level runner for figure 1
    :return: None
    """
    print("Running: Figure 1")

    init_figure(size=(7, 3))
    plt.subplot2grid((2, 6), (0, 0), colspan=2, rowspan=1)
    figure_1a("A")

    plt.subplot2grid((2, 6), (0, 2), colspan=2, rowspan=1)
    figure_1b("B")

    plt.subplot2grid((2, 6), (0, 4), colspan=2, rowspan=1)
    figure_1c("C")

    for ix, col in enumerate([0, 3]):
        plt.subplot2grid((2, 6), (1, col), colspan=3, rowspan=1)
        figure1d("D" + str(ix + 1), panel=ix)

    save_fig("1")

    figure1_supplemental_files()


def figure1_supplemental_files():
    """Make supplemental figures using unmodified parameters taken directly from paper data"""
    """
    If we take the paper's value of 5.92 nS and a 1cm^2 patch is equivalent to 5.92/1e6 mS/cm^2
    This gives a plot scaled to almost 0 because this is a negligible effective channel density
    """
    paper_value = 5.92 / 1e6  # nS/cm^2 in mS/cm^2
    figure_1a("Supplemental 1A(v1)", g_na=paper_value)
    plt.ylim([-1, 1])  # reset to make line visible
    save_fig("supp_1A_v1")

    """
    If we assume the paper mis-reported the number 5.92mS/cm^2 as 5.92nS/cm^2 then we get a plot scaled by ~2
    The "onset of current" at ~-50 mV and the reversal potential at ~60mV is preserved
    """
    figure_1a(
        "Supplemental 1A(v2)", g_na=5.92
    )  # assume units incorrect and meant mS/cm^2
    save_fig("supp_1A_v2")

    """
    Similarly to figure 1A they report 9.12nS/cm^2 again we will assume this 9.12nS but 9.12mS/cm^2 additionally if we
    use a pule width of 3ms as reported we see the current is far too large
    """
    figure_1b("Supplemental 1B(v1)", g_na=9.12, pulse_width=3)
    save_fig("supp_1B_v1")

    """
    If we adjust g_na and leave the pulse width at 3ms we get an appropriate current scale but the current does not 
    decay enough.
    """
    figure_1b("Supplemental 1B(v2)", pulse_width=3)
    save_fig("supp_1B_v2")

    # If we leave g_na at 9.12mS/cm^2 and set the pulse width to 5ms we still see a current that's ~2x too large
    figure_1b("Supplemental 1B(v3)", g_na=9.12)
    save_fig("supp_1B_v3")

    """
    If we use the original equation for tau_n (with the shift being 40mV) then we see a very distorted limit cycle in 
    the n,h phase space
    """
    figure_1c("Supplemental 1C", use_modified_tau_n=False)
    save_fig("supp_1C")

    """
    If we use the original equation for tau_n (with the shift being 40mV) then we see a very distorted voltage trace
    Incidentally this looks more like the GABA neuron in their cited Seutin and Engel paper (2010) with a rounded trough
    see figure 7B
    """
    figure1d("Supplemental 1D2", panel=1, use_modified_tau_n=False)
    save_fig("supp_1D2")


def figure_1a(title, g_na=5.92 * 0.514, v_reset=-120):
    """Compute IV curve for 2d and 3d ode model

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
    # Compute an initial condition that is the model clamped at v_reset
    initial_condition = steady_state_when_clamped(v_reset)

    # Perform the same voltage clamp experiment for multiple models
    for model in [ode_3d, ode_2d]:
        initial_condition = resize_initial_condition(initial_condition, model)

        # Set plot properties for the model
        color, linestyle = ("grey", "solid") if model is ode_3d else ("black", "--")

        # compute IV curve
        current, voltage = current_voltage_curve(
            model=model,
            clamp_range=[-90, 60],
            t_max=500,
            ic=initial_condition,
            g_na=g_na,
        )
        # plot IV curve
        plt.plot(voltage, current, color=color, linestyle=linestyle)

    # plot settings
    plt.legend(["3D", "2D"], loc="center left", bbox_to_anchor=(0.3, 1.05))
    set_properties(
        title,
        x_label="V (mV)",
        y_label="I$_{Na}$($\mu$A/cm$^2$)",
        x_tick=[-80, -40, 0, 40],
        y_tick=[-160, 0],
    )


def generate_clamp_pattern_1b(t_max, pulse_width=5):
    """Helper function to generate the waveform for figure 1b

    Figure 1b requires a clamp waveform with timing requirements that are annoying to generate this function
    encapsulate that operation

    :param t_max: End of the simulation
    :param pulse_width: Optional width to the pulse (ms): defaults to 5ms
    :return: Returns the clamp pattern in the "pattern" format
    """
    # make pulses 100ms and the pulse transitions from between -70mV to 0mV
    pulse_period = 100
    clamp_potential = {"low": -70, "high": 0}

    # initialize clamped low and end current regime at t=0
    segment_start = [-pulse_period]
    segment_end = [0]
    voltage_sequence = [clamp_potential["low"]]
    clamped_high = False
    # Generate start and end times for the waveform until the simulation end is hit
    while segment_end[-1] < t_max:

        # If the membrane is currently clamped low: compute pulse time and set new clamp parameters
        segment_start.append(segment_end[-1])
        if not clamped_high:
            remaining_time = pulse_width
            voltage_sequence.append(clamp_potential["high"])
            clamped_high = True
        else:
            remaining_time = pulse_period - pulse_width
            voltage_sequence.append(clamp_potential["low"])
            clamped_high = False
        # end "remaining time" after current start time
        segment_end.append(segment_start[-1] + remaining_time)

    # Return a dictionary of start_time:clamp_potential key-value pairs
    return {k: v for k, v in zip(segment_start, voltage_sequence)}


def figure_1b(title, g_na=5.92, pulse_width=5):
    """Compute the periodic step current response from figure 1B

    Clamp to membrane potential to -80 then depolarize and rest the membrane potential to 0mV and -70mV every 100ms with
    5mV pulses to 0mV.

    In the original paper g_na is 9.12 mS/cm^2 and pulses are 5ms to reproduce their results we use g_na = 5.92 and a
    5 ms pulse

    :param title: Plot title (panel label)
    :param g_na:  Optional sodium conductance to use: defaults to working parameter
    :param pulse_width:  Optional pulse width (ms) to use: defaults to working parameter
    :return: None
    """
    # Create a 500ms simulation clamped at -80 which then goes through the 1b clamp pattern
    t_max = 500
    pattern = generate_clamp_pattern_1b(t_max, pulse_width=pulse_width)
    initial_condition = steady_state_when_clamped(v_clamp=-80)

    # Impose v_clamp according to pattern
    solution, time, waveform = pulse(
        model=ode_3d,
        parameter_name="v_clamp",
        temporal_pattern=pattern,
        t_max=t_max,
        ic=initial_condition,
        g_na=g_na,
    )

    # Compute sodium current and plot
    i_na = sodium_current(solution.T, default_parameters(g_na=g_na))

    # Plot Na function
    plt.plot(time, i_na, "k")

    # Plot properties
    set_properties(
        title,
        x_label="Time (ms)",
        y_label="",  # this is actually uA/cm^2 todo discuss in paper
        x_tick=[0, 200, 400],
        y_tick=[-250, -200, 0],
        x_limits=[-50, 450],
    )


def figure_1c(title, use_modified_tau_n=True):
    """Compute limit cycle in n,h phase space for the 5d model and compute the approximation n=f(h) for 1C

    :param title: Plot title (panel label)
    :param use_modified_tau_n: Optional parameter to use the original tau_n which does not work. Defaults to our tau_n
    :return: None
    """
    initial_condition = [
        -55,
        0,
        0,
        0,
        0,
    ]  # Does not need to lie on limit cycle since we throw away transient

    # Solve 5d model with appropriate tau_n
    partial_5d = partial(ode_5d, use_modified_tau_n=use_modified_tau_n)
    t, sol = solve_ode(partial_5d, initial_condition, t_max=4200, dt=0.1, rtol=1e-3)

    # Extract h and n and discard the first half due to transient
    ix_half_time = int(len(t) / 2)
    h = sol[ix_half_time:, 1]
    n = sol[ix_half_time:, 4]
    fit_f_approx(h, n)

    # Plot limit cycle in phase plane
    plt.plot(h, n, c="grey")
    plt.plot(h, f_approx(h), "k")

    # Plot properties
    plt.legend(["n", "n=f(h)"], loc="center left", bbox_to_anchor=(0.3, 1.05))
    set_properties(
        title,
        x_label="h",
        y_label="n",
        x_tick=[0, 0.2, 0.4, 0.6],
        y_tick=np.arange(0, 1, 0.2),
        x_limits=[0, 0.7],
    )


def fit_f_approx(h, n):
    replicate_fit = np.poly1d(np.polyfit(h, n, deg=3))
    print(replicate_fit)


def figure1d(title, panel=0, use_modified_tau_n=True):
    """Show waveforms for 3d (ix=0) or 5d (ix=1) models

    :param title: Plot title (panel label)
    :param panel: Set the model to use 3d/5d (panel=0/1)
    :param use_modified_tau_n: Optional parameter to use the original tau_n which does not work. Defaults to our tau_n
    :return: None
    """
    # Select appropriate model depending on the panel
    model = [ode_3d, ode_5d][panel]

    # Run the simulation for 4200ms and start at an arbitrary point "close" to the limit cycle
    initial_condition = [-55, 0, 0]
    initial_condition = resize_initial_condition(initial_condition, model, fill=0)

    # Solve 5d model with appropriate tau_n
    if model == ode_5d:
        model = partial(ode_5d, use_modified_tau_n=use_modified_tau_n)

    t, sol = solve_ode(model, initial_condition, t_max=4200, dt=0.1, rtol=1e-3)

    # Throw away the first 1000ms of the simulation
    t_throw_away = np.where(t > 1000)[0][0]
    sol = sol[t_throw_away:, :]
    t = t[t_throw_away:] - t[t_throw_away]  # set new t[0]=0

    # Plot voltage trace
    plt.plot(t, sol[:, 0], "k")
    y_label = "V (mV)" if panel == 0 else ""
    y_tick_label = None if panel == 0 else []

    # Plot properties
    set_properties(
        title,
        x_label="Time (ms)",
        y_label=y_label,
        y_tick=[-80, -40, 0],
        y_limits=[-80, 20],
        y_ticklabel=y_tick_label,
        x_tick=[0, 1000, 2000, 3000],
        x_limits=[0, 3000],
    )
