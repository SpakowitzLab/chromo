"""Utilities for adapting Monte Carlo amplitudes."""

import os
import sys

import numpy as np
import pandas as pd


class FOPTD_PID(object):
    """
    Class representation of a PID controller for closed-loop feedback control.
    """

    def __init__(self, setpoint, Kc, tau_i, tau_d):
        """
        Initialize PID controller with tuning parameters.

        Parameters
        ----------
        setpoint : float
            Setpoint of controller
        Kc : float
            Controller gain
        tau_i : float
            Integral term time constant
        tau_d : float
            Derivative term time constant
        """
        self.setpoint = setpoint
        self.Kc = Kc
        self.tau_i = tau_i
        self.tau_d = tau_d
        self.time = {}
        self.int_err = {}
        self.err = {}
        self.control_amp_bead = True
        self.control_amp_move = True
    
    def PID_Step(self, adapter_name, time, data_point):
        """
        Response of PID controller to new data point.

        Parameters
        ----------
        adapter_name : str
            Name of the MC adapter affected by the controller.
        time : float
            Time point of new data measurement
        data_point : float
            New data point to be controlled to setpoint.
        
        Returns
        -------
        output : float
            Controller output
        """
        err = data_point - self.setpoint

        # Check if first step.
        if len(self.time[adapter_name]) < 1:
            self.time[adapter_name].append(time)
            self.err[adapter_name].append(err)
            return 0

        # Identify time step
        prev_time = self.time[adapter_name][-1]
        dt = time - prev_time
        self.time[adapter_name].append(time)
        
        # Identify process error (deviation from setpoint)
        prev_err = self.err[adapter_name][-1]
        self.err[adapter_name].append(err)
        
        # Approximate integral of error with trapezoid
        step_int_err = (prev_err + 0.5 * (err - prev_err)) * dt
        self.int_err[adapter_name] += step_int_err

        # Approximate derivative of the error term
        de_dt = (err - prev_err) / dt

        P = self.Kc * err
        I = self.Kc / self.tau_i * self.int_err[adapter_name]
        D = self.Kc * self.tau_d * de_dt
        output = P + I + D

        return output

    def control_mc(self, mc_adapter):
        """
        Apply PID controller to new MC step.

        Parameters
        ----------
        mc_adapter : MCAdapter Object
            MC move parameters for adaption based on move acceptance rate

        Returns
        -------
        mc_adapter : MCAdapter Object
            Updated MC move parameters based on PID controller output
        """
        mc_adapter, _, acceptance_rate, _, _ = performance_to_file(mc_adapter)

        if mc_adapter.name not in self.time:
            self.time[mc_adapter.name] = []
            self.err[mc_adapter.name] = []
            self.int_err[mc_adapter.name] = 0

        # Check if controller is active
        if self.control_amp_bead or self.control_amp_move:
            time = mc_adapter.performance_tracker.all_steps[-1]
            output = self.PID_Step(mc_adapter.name, time, acceptance_rate)
        else:
            return mc_adapter
        
        if not np.isclose(output, 0):
            if self.control_amp_bead:
                mc_adapter = self.check_limits(mc_adapter, output, "amp_bead", "bead_amp_range")
            if self.control_amp_move:
                mc_adapter = self.check_limits(mc_adapter, output, "amp_move", "move_amp_range")
        
        return mc_adapter

    def check_limits(self, mc_adapter, output, attribute, limit_attribute=None):
        """
        Make sure amplitude does not exceed their limits.

        Parameters
        ----------
        mc_adapter : MCAdapter Object
            MC move parameters for adaption based on move acceptance rate
        output : float
            Output of PID controller based on observed acceptance rate
        attribute : str
            mc_adapter attribute name containing parameter being controlled
        limit_attribute : str (optional)
            mc_adapter attribute name containing limits in list of length 2;
            If None, no adjustment for limits is applied to the parameter

        Returns
        -------
        mc_adapter : MCAdapter Object
            Updated MC move parameters based on PID controller output
        """
        limits = getattr(mc_adapter, limit_attribute)
        prev_val = getattr(mc_adapter, attribute)
        new_val = prev_val + output
        
        setattr(mc_adapter, attribute, new_val)

        if limit_attribute is not None:
            
            if output < 0:
                if new_val < limits[0]:
                    
                    self.int_err[mc_adapter.name] -= \
                        self.err[mc_adapter.name][-1] * \
                        (self.time[mc_adapter.name][-1] - 
                        self.time[mc_adapter.name][-2])

                    setattr(mc_adapter, attribute, limits[0])

            elif output > 0:
                if new_val > limits[1]:

                    self.int_err[mc_adapter.name] -= \
                        self.err[mc_adapter.name][-1] * \
                        (self.time[mc_adapter.name][-1] - 
                        self.time[mc_adapter.name][-2])

                    setattr(mc_adapter, attribute, limits[1])

        return mc_adapter


class PerformanceTracker(object):
    """
    Class representation of performance tracker for MC adapter.
    
    Tracks the acceptance rate by monitoring the fraction of the N most recent
    moves (specified by `num_steps_tracked`) to be accepted. Logs the 
    acceptance rate, move amplitude, and bead amplitude at each step after a 
    startup period given by `startup_steps`.

    This class is used to provide metrics to the MC adaptor with which changes
    to move and bead amplitudes can be applied.
    """

    def __init__(self, num_steps_tracked=50, startup_steps=100):
        """
        Initialize feedback controller for MC adapter.

        Parameters
        ----------
        num_steps_tracked : int
            Number of preceeding steps to factor into calculation of current
            move acceptance rate
        startup_steps : int
            Number of steps to include in the startup of the MC simulator,
            to generate a baseline move acceptance rate before adaption is
            applied. Must be greater than `num_steps_tracked` or is replaced by
            `num_steps_tracked`.
        """
        self.startup = True
        self.step_count = 0
        self.num_steps_tracked = num_steps_tracked
        self.tracked_steps = [i%2 for i in range(num_steps_tracked)]
        self.startup_steps = max(num_steps_tracked, startup_steps)
        self.acceptance_rate = None
        self.acceptance_history = []
        self.amp_bead_history = []
        self.amp_move_history = []
        self.steps = []
        self.all_steps = []
    
    def redo_startup(self):
        """
        Redo the startup tracking of move acceptance rates.

        This method allows you to change the number of tracked steps in the
        simulation.
        """
        self.tracked_steps = [None] * self.num_steps_tracked
        self.step_count = 0
        self.startup = True
        
        if self.num_steps_tracked > self.startup_steps:
            self.startup_steps = self.num_steps_tracked
    
    def add_step(self, accepted):
        """
        Add an accepted (1) or rejected (2) step to the log.

        Parameters
        ----------
        accepted : bool
            Binary indicator for whether a move was accepted (True) or
            rejected (False)
        """
        self.tracked_steps.insert(0, int(accepted))
        self.tracked_steps.pop()
        
        if self.step_count > self.startup_steps:
            self.startup = False

    def calc_acceptance_rate(self):
        """
        Calculate acceptance rate among `num_steps_tracked` recent moves.
        
        Returns
        -------
        float
            Fraction of recent MC moves which were accepted
        """
        step_accept = [x for x in self.tracked_steps if x is not None]
        return sum(step_accept) / len(step_accept)
    
    def log_performance(self, step, acceptance_rate, amp_bead, amp_move):
        """
        Record statistics of MC simulator performance.

        Parameters
        ----------
        step : int
            Current step in MC process
        acceptance_rate : float
            Fraction of recent moves accepted by MC simulator
        amp_bead : int
            Maximum number of beads affected by an MC move in a single step
        amp_move : float
            Maximum amplitude of a MC move in a single step
        """
        self.acceptance_history.append(acceptance_rate)
        self.amp_bead_history.append(amp_bead)
        self.amp_move_history.append(amp_move)
        self.steps.append(step)
    
    def save_performance(self, file_path):
        """
        Save history of MC move acceptance to a CSV file.

        Save the history of accepted MC moves, bead amplitudes, and move
        amplitudes to a log file at the specified file location. If a file
        already exists at the specified file location, the file will be
        APPENDED, not overwritted. After logging the performance metrics,
        the history of those performance metrics will be cleared. 

        Parameters
        ----------
        file_path : str
            Path to file logging move acceptance history
        """
        performance_history = np.column_stack((
            np.array(self.steps),
            np.array(self.acceptance_history),
            np.array(self.amp_bead_history),
            np.array(self.amp_move_history)
        ))
        performance_history.round(decimals=6, out=performance_history)
        
        if os.path.exists(file_path):
            column_labels = ""
        else:
            column_labels = "steps\tacceptance\twindow_size\tmove_amp"

        f = open(file_path, 'ab')
        np.savetxt(f, performance_history, fmt="%1.6f", delimiter="\t", 
            newline="\n", header=column_labels)
        f.close()

        self.steps = []
        self.acceptance_history = []
        self.amp_bead_history = []
        self.amp_move_history = []

    def store_performance(self, amp_bead, amp_move):
        """
        Get current acceptance rate from `PerformanceTracker` & add to log.

        Parameters
        ----------
        amp_bead : int
            Maximum number of beads affected by a single MC step
        amp_move : float
            Maximum amplitude of a move in a single MC step
        """
        step = self.step_count
        acceptance_rate = self.calc_acceptance_rate()
        self.log_performance(step, acceptance_rate, amp_bead, amp_move)


def basic_feedback_adaption(mc_adapter):
    """
    Adjust bead and move amplitudes based on move acceptance.

    Calculate the acceptance rate of the MC move based on attempt and success
    counts. Check the acceptance rate relative to a setpoint and upper/lower
    bounds. If the acceptance rate is not equal to the setpoint and falls
    between the upper and lower bounds, adjust the max number of beads affected 
    by the move in the next iteration. If the acceptance rate falls outside the
    bounds, adjust the maximum amplitude of the move in the next iteration.

    Parameters
    ----------
    mc_adapter : MCAdapter Object
        MC move parameters for adaption based on move acceptance rate
    
    Returns
    -------
    MCAdapter Object
        MC move parameters updated based on move acceptance rate
    """
    mc_adapter, _, acceptance_rate, _, _ = performance_to_file(mc_adapter)

    lower_lim = mc_adapter.thresholds[0]
    setpoint = mc_adapter.thresholds[1]
    upper_lim = mc_adapter.thresholds[2]

    if np.isclose(acceptance_rate, setpoint):
        return mc_adapter
    
    adapt_case = np.digitize(acceptance_rate, 
        [-np.inf, lower_lim, setpoint, upper_lim, np.inf])
    
    if adapt_case == 1:
        # acceptance rate way too low; decrease move amplitude
        return adjust_amp_move(mc_adapter, acceptance_rate, decrease=True)
    elif adapt_case == 2:
        # acceptance rate slightly low; decrease bead amplitude
        return adjust_amp_bead(
            mc_adapter, acceptance_rate, decrease = True)
    elif adapt_case == 3:
        # acceptance rate slightly high; decrease bead amplitude
        return adjust_amp_bead(
            mc_adapter, acceptance_rate, decrease = False)
    else:
        # acceptance rate way too high; increease move amplitude
        return adjust_amp_move(mc_adapter, acceptance_rate, decrease=False)


def deviation_from_setpoint(mc_adapter, acceptance_rate):
    """
    Quanttify the deviation of MC acceptance rate from setpoint.
    
    Parameters
    ----------
    mc_adapter : MCAdapter Object
        MC move parameters for adaption based on move acceptance rate
    acceptance_rate : float [0, 1]
        Fraction of MC move attempts which are accepted

    Returns
    -------
    deviation : float
        Absolute difference between move acceptance rate and setpoint
    max_deviation : float
        Maximum possible difference between move acceptance rate and setpoint
    """
    setpoint = mc_adapter.thresholds[1]
    deviation = np.absolute(acceptance_rate - setpoint)
    max_deviation = max(setpoint, 1-setpoint)
    return deviation, max_deviation


def adjust_amp_move(mc_adapter, acceptance_rate, decrease=True):
    """
    Adjust the MC move amplitude.

    Increase or decrease the MC move amplitude by multiplying or dividing the
    current value, respectively, with a multiplicative factor between 0 and 1.
    
    The multiplicative factor is set based on the deviation in move acceptance
    rate from the setpoint. When move acceptance rates are close to the
    setpoint, multiply or divide the move amplitude by a multiplicative factor
    closer to 1. When move acceptance rates are far from the setpoint, multiply
    pr divide the move amplitude by a value further from one.

    Parameters
    ----------
    mc_adapter : MCAdapter Object
        MC move parameters for adaption based on move acceptance rate
    acceptance_rate : float [0, 1]
        Fraction of MC move attempts which are accepted
    decrease : bool (optional, default = True)
        Flag for whether to increase or decrease move amplitude
    
    Returns
    -------
    MCAdapter Object
        MC move parameters updated based on move acceptance rate
    """
    deviation, max_deviation = deviation_from_setpoint(
        mc_adapter, acceptance_rate)
    
    factor = np.interp(deviation, [0, max_deviation], 
        mc_adapter.move_factor[::-1])

    if decrease == True:
        mc_adapter.amp_move = max((mc_adapter.amp_move * factor),
            mc_adapter.move_amp_range[0])
    else:
        mc_adapter.amp_move = min((mc_adapter.amp_move / factor), 
            mc_adapter.move_amp_range[1])

    return mc_adapter


def adjust_amp_bead(mc_adapter, acceptance_rate, decrease=True):
    """
    Adjust the MC bead amplitude.

    See documentation for `adjust_amp_move` for general approach.

    Parameters
    ----------
    mc_adapter : MCAdapter Object
        MC move parameters for adaption based on move acceptance rate
    acceptance_rate : float [0, 1]
        Fraction of MC move attempts which are accepted
    decrease : bool (optional, default = True)
        Flag for whether to increase or decrease bead amplitude
    
    Returns
    -------
    MCAdapter Object
        MC move parameters updated based on move acceptance rate
    """
    deviation, max_deviation = deviation_from_setpoint(
        mc_adapter, acceptance_rate)
    
    factor = np.interp(deviation, [0, max_deviation], 
        mc_adapter.window_factor[::-1])

    if decrease == True:
        mc_adapter.amp_bead = max((mc_adapter.amp_bead * factor),
            mc_adapter.bead_amp_range[0])
    else:
        mc_adapter.amp_bead = min((mc_adapter.amp_bead / factor),
            mc_adapter.bead_amp_range[1])
    
    return mc_adapter


def performance_to_file(mc_adapter):
    """
    Log the performance of the MC adapter to an output file.

    Parameters
    ----------
    mc_adapter : MCAdapter Object
        MC move parameters for adaption based on move acceptance rate
    
    Returns
    -------
    mc_adapter : MCAdapter Object
        MC adapter with updated performance log
    step : int
        Step of simulation
    acceptance_rate : float
        Fraction of recent MC steps accepted
    amp_bead : int
        Maximum number of beads affected by a single MC step
    amp_move : float
        Maximum size of an MC move applied at a single step
    """
    step = mc_adapter.performance_tracker.step_count
    acceptance_rate = mc_adapter.performance_tracker.calc_acceptance_rate()
    amp_bead = mc_adapter.amp_bead
    amp_move = mc_adapter.amp_move

    mc_adapter.performance_tracker.log_performance(
        step, acceptance_rate, amp_bead, amp_move)

    return mc_adapter, step, acceptance_rate, amp_bead, amp_move