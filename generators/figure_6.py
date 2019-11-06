from scipy.integrate import odeint

from helpers.plotting import *
from ode_functions.current import nmda_current, ampa_current
from ode_functions.diff_eq import synaptic_3d, default_parameters
from ode_functions.gating import *


def run():
    """
    Top level runner for figure 6
    :return: None
    """

    init_figure(size=(6, 6))
    __figure6__()

    save_fig('6')


def __figure6__():
    e_syn = 0  # todo: is this correct?
    stimulus_types = {'nmda': nmda_current, 'ampa': ampa_current, 'i': None}
    parameter_sets_a = {'nmda': [0, 0.060, 0], 'ampa': [0, 0.0023, 0], 'i': [0, 0.16, 0]}  # a parameters
    parameter_sets_b = {'nmda': [0, 0.060, 0], 'ampa': [0, 0.0007, 0], 'i': [0, 0.32, 0]}  # b parameters
    all_parameter_sets = [parameter_sets_a, parameter_sets_b]
    times = [2000, 8000, 10000]  # onset, offset, end times

    for iz, parameter_sets in enumerate(all_parameter_sets):  # iterate over fig a/b parameter sets
        for ix, perturbation_type in enumerate(['nmda', 'ampa', 'i']): # iterate over type of channel nmda, ampa, inj.

            parameter_set = parameter_sets[perturbation_type]
            stimulus_type = stimulus_types[perturbation_type]

            t0 = 0
            ic = [-65, 1, 1]
            t_solved = np.array([])
            solution = np.array([0, 0, 0])

            plt.subplot(3, 2, 2 * ix + iz + 1)
            # if there is a synapse defined: control parameter is g_syn, otherwise it's i_app
            for iy, parameter_value in enumerate(parameter_set): # iterate over the value of the channel for pulse

                t = np.arange(t0, times[iy])
                t_solved = np.concatenate((t_solved, t))
                t0 = times[iy]

                if stimulus_type is not None: # there is a channel
                    parameters = default_parameters(i_app=0)
                    parameters['g_syn'] = parameter_value
                    parameters['e_syn'] = e_syn
                else:
                    parameters = default_parameters(i_app=parameter_value)

                state = odeint(synaptic_3d, ic, t, args=(parameters, stimulus_type))

                ic = state[-1, :] # maintain the initial condition of the next time step

                solution = np.vstack((solution, state))

            solution = solution[1:, :]  # first row is [0,0] for starting shape so omit

            plt.plot(t_solved, solution[:, 0], 'k')

            # plot setting generation - not germaine to simulations
            title = "A"
            y_label = "V (mV)"
            y_ticklabel = None
            if iz == 1:
                y_label = ""
                y_ticklabel = []
                title = "B"
            title += str(ix + 1)
            x_label = ""
            x_ticklabel = []
            if (2 * ix + iz + 1) == 5 or (2 * ix + iz + 1) == 6:
                stimulus = np.zeros(t_solved.shape)
                stimulus[(t_solved > times[0]) & (t_solved < times[1])] = 1

                plt.plot(t_solved, 10 * stimulus - 80, 'grey')
                x_label = "t (ms)"
                x_ticklabel = None

            set_properties(title, y_label=y_label, y_tick=[-80, -40, 0], y_ticklabel=y_ticklabel,
                           x_tick=[0, 5000, 10000],
                           x_label=x_label, x_ticklabel=x_ticklabel)
