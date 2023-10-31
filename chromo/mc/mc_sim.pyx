# cython: profile=True

"""Routines for performing Monte Carlo simulations.
"""

import pyximport
pyximport.install()

# Built-in Modules
from typing import List, TypeVar, Optional
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp
#from time import process_time
#import warnings
#import sys

# External Modules
import numpy as np

# Custom Modules
from chromo.polymers import PolymerBase
from chromo.polymers cimport PolymerBase
from chromo.binders import ReaderProtein
from chromo.mc.moves import MCAdapter
from chromo.mc.moves cimport MCAdapter
from chromo.mc.mc_controller import Controller
from chromo.fields cimport UniformDensityField as Udf
from chromo.polymers import SSTWLC


cpdef void mc_sim(
    list polymers, readerproteins, long num_mc_steps,
    list mc_move_controllers, Udf field, double mu_adjust_factor,
    long random_seed, double temperature_adjust_factor, double lt_value_adjust
):
    """Perform Monte Carlo simulation.

    Pseudocode
    ----------
    Repeat for each Monte Carlo step:
        Repeat for each adaptable move:
            If active, apply move to each polymer
            
    Notes
    -----
    To report process time for each interval of MC steps, add the following:
    
    if (k+1) % 50000 == 0:
        print("MC Step " + str(k+1) + " of " + str(num_mc_steps))
        print(
            "Time for previous 50000 MC Steps (in seconds): ", round(
                process_time()-t1_start, 2
            )
        )
        t1_start = process_time()

    Parameters
    ----------
    polymers : List[Polymer]
        Polymers affected by Monte Carlo simulation
    readerproteins : List[ReaderProteins]
        Specification of reader proteins on polymer
    num_mc_steps : int
        Number of Monte Carlo steps to take between save points
    mc_move_controllers : List[Controller]
        List of controllers for active MC moves in simulation
    field: UniformDensityField
        Field affecting polymer in Monte Carlo simulation
    mu_adjust_factor : double
        Adjustment factor applied to the chemical potential in response to
        simulated annealing
    random_seed : Optional[int]
        Randoms seed with which to initialize simulation 
    """
    cdef bint active_field
    cdef long i, j, k, n_polymers
    cdef list active_fields
    cdef poly
    np.random.seed(random_seed)
    if field.confine_type == "":
        active_fields = [poly.is_field_active() for poly in polymers]
    else:
        active_fields = [1 for _ in polymers]
    for k in range(num_mc_steps):
        print("a")
        for controller in mc_move_controllers:
            print("b")
            if controller.move.move_on == 1:
                print("c")
                for j in range(controller.move.num_per_cycle):
                    print("d")
                    for i in range(len(polymers)):
                        print("e")
                        #polymers[i].lt = lt
                        polymers[i].lt = lt_value_adjust
                        polymers[i]._find_parameters(polymers[i].bead_length) # coding best practice: accessing protected member of a class from outside the class?
                        poly = polymers[i]
                        active_field = active_fields[i]
                        print("before mc step")
                        #controller.move = MCAdapter.specific_moves.end_pivot
                        print("does printing work here")
                        print(controller.move)
                        print(poly)
                        print(readerproteins)
                        print(field)
                        print(active_field)
                        mc_step(
                            controller.move, poly, readerproteins, field,
                            active_field, temperature_adjust_factor
                        )
                        print("after mc step")
            controller.update_amplitudes()


cpdef void mc_step(
    MCAdapter adaptible_move, PolymerBase poly, readerproteins,
    Udf field, bint active_field, double temperature_adjust_factor
):
    """Compute energy change and determine move acceptance.

    Notes
    -----
    Get the proposed state of the polymer. Calculate the total (polymer +
    field) energy change associated with the move. Accept or reject the move
    based on the Metropolis Criterion.

    After evaluating change in energy (from the polymer and the field), the
    try-except statement checks for RuntimeWarning if the change in energy gets
    too large.

    Parameters
    ----------
    adaptible_move: MCAdapter
        Move applied at particular Monte Carlo step
    poly: PolymerBase
        Polymer affected by move at particular Monte Carlo step
    readerproteins: List[ReaderProteins]
        Reader proteins affecting polymer configuration
    field: UniformDensityField
        Field affecting polymer in Monte Carlo step
    active_field: bool
        Indicator of whether or not the field is active for the polymer
    temperature_adjust_factor: double
        Divide the energy value used to select accepted moves by this factor
    """
    cdef double dE, exp_dE
    cdef int check_field = 0
    cdef long packet_size, n_inds
    cdef long[:] inds


    packet_size = 1 # will need to change back
    print("entering mc")
    if poly in field and active_field:
        print("this is it probably")
        if adaptible_move.name != "tangent_rotation":
            print("interesting")
            check_field = 1
    packet_size = 20
    inds = adaptible_move.propose(poly)
    n_inds = len(inds)
    print("intermediate test")
    if n_inds == 0:
        print("is this it")
        return

    dE = 0
    dE += poly.compute_dE(adaptible_move.name, inds, n_inds)
    if check_field == 1:
        print("field 1")
        if adaptible_move.name == "change_binding_state":
            print("additional test location")
            dE += field.compute_dE(
                poly, inds, n_inds, packet_size, state_change=1
            )
        else:
            print("field 0")
            dE += field.compute_dE(
                poly, inds, n_inds, packet_size, state_change=0
            )
    try:
        print("energy exp")
        exp_dE = exp(-dE)
    except RuntimeWarning:
        if dE > 0:
            exp_dE = 0
        elif dE < 0:
            exp_dE = 1

    if (<double>rand() / RAND_MAX) < exp_dE/temperature_adjust_factor: #temperature variation
        #print(<double>rand() / RAND_MAX)
        adaptible_move.accept(
            poly, dE, inds, n_inds, log_move=True, log_update=True
        )
        if check_field == 1:
            field.update_affected_densities()

    else:
        adaptible_move.reject(
            poly, dE, log_move=True, log_update=True
        )

