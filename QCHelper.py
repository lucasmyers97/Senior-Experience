# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:30:44 2019

@author: Lucas
"""

import numpy as np
from scipy.linalg import qr
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
from pyquil.quil import Program

def getBinFrac(frac, num_digits):
    """
    Take in a fraction, and spit out the representation in binary.
    
    Works by successively multiplying the fraction by 2: if the number
    becomes greater than 1, a 1 is added to the binary representation,
    if the number is still less than 1, a 0 is added to the binary
    representtion. 
    
    Parameters
    ----------
    frac: decimal < 1
        Fraction that will be converted into binary.
    num_digits: integer
        Number of digits in final binary representation.
        
    Returns
    -------
    bin_repr: string
        String representing the binary representation of the 
        input fraction.
        
    Examples
    --------
    >>> getBinFrac(1/2, 3)
    '0.100'
    >>> getBinFrac(1/16, 5)
    '0.00010'
    >>> getBinFrac(1/13, 10)
    '0.0001001110'
    
    """
    
    bin_repr = "0."
    for i in range(num_digits):
        frac *= 2
        if frac >= 1:
            frac -= 1
            bin_repr += '1'
        else:
            bin_repr += '0'
            
    return bin_repr

def genRandomVector(k, zero_places=0, rand_seed=42):
    """
    Return a random, normalized vector over the complex field.
    
    The vector is length-k. You may specify the number of places at the
    beginning of the vector which are forced to be zero. This is helpful when 
    testing the arbitrary n-qubit tranformation algorithm. Additionally, you
    can specify a random seed to make sure you can reuse the random vector if
    it identifies a bug somewhere.
    
    Parameters
    ----------
    k: integer
        Length of the random vector.
    zero_places: integer
        Number of places at the beginning of the random vector which are zero.
    rand_seed: integer
        Seed used to generate the random phases and amplitudes.
        
    Returns
    -------
    norm_rand_state: ndarray
        Length-k complex vector which is normalized.
    """
    
    np.random.seed(seed=rand_seed)
    
    # Create random complex state which is in gray code ordering
    rand_amps = np.random.rand(k)
    rand_phases = np.exp( 1j*np.random.rand(k)*2*np.pi )
    rand_state = rand_amps*rand_phases
    rand_state[0:zero_places] = 0
    norm_rand_state = rand_state / np.linalg.norm(rand_state)
    
    return norm_rand_state

def genRandomUnitary(k, rand_seed=42):
    """
    Return random matrix of size k x k using QR decomposition.
    
    Parameters
    ----------
    k: integer
        Size of one dimension of the random unitary matrix matrix.
    rand_seed: integer
        Seed used to generate random unitary. 
    """
    
    np.random.seed(seed=rand_seed)
    
    # Generate matrix with random amplitude and phase
    amplitude = np.random.rand(k, k)
    phase = np.exp( (1j*2*np.pi)*np.random.rand(k, k) )
    random_matrix = amplitude*phase
    
    # Extract unitary using QR decomposition
    random_unitary, throwaway_matrix = qr(random_matrix)
    
    return random_unitary

def getMatFromProgram(n_qubits, p):
    """
    Return matrix (in standard ordering) which represents the effect of p on
    a wavefunction.
    
    Note that you must have a quil compiler and a quantum virtual machine
    running on your computer in order for this function to execute 
    successfully.
    
    Parameters
    ----------
    n_qubits: integer
        Number of qubits in the register which the Program `p` acts on.
    p: pyQuil Program
        Program for which you are trying to find a matrix representation
        
    Returns
    -------
    A: ndarray, dtype=np.complex_
        Complex unitary matrix representing the effect of Program `p` on a
        state.
    """
    
    A = np.zeros( (2**n_qubits, 2**n_qubits), dtype=np.complex_ )
    
    # For each basis qubit, see where p sends it
    for i in range(2**n_qubits):
        bit_string = [int(j) for j in np.binary_repr(i, width=n_qubits)]
        bit_string.reverse()
        
        # Prep state so it looks like current bit string
        state_prep_p = Program()
        for j in range(n_qubits):
            if bit_string[j] == 1:
                state_prep_p.inst( Program(X(j)) )
            else:
                state_prep_p.inst( Program(I(j)) )
                
        # See what p does to state, record result in matrix A
        wfn = WavefunctionSimulator().wavefunction( state_prep_p.inst(p) )
        A[:, i] = wfn.amplitudes[0:2**n_qubits]
        
    return A

def arbitraryGate(Q):
    """
    Return numpy array which represents the single-qubit operation encoded by
    alpha, beta, delta, and gamma in the dictionary `Q`.
    
    In Rieffel and Polak, they show that an arbitrary single-qubit
    transformation can be uniquely defined by four real numbers. This function
    uses the same convention, naming these numbers `alpha`, `beta`, `delta`, 
    and `gamma`, and using a dictionary `Q` to hold each number indexed by the 
    same name. The purpose of this function is to test the actual encoding of
    the arbitrary transformation into parametric gates from Rigetti Forest.
    
    Parameters
    ----------
    Q: dictionary
        Holds the four angles `alpha`, `beta`, `delta`, and `gamma` used to
        specify a single-qubit transformation by the Rieffel and Polak
        convention. These angles are indexed by their names.

    Returns
    -------
    A: np.ndarray
        2x2 matrix representing the transformation encoded in `Q`.
    """
    
    if 'alpha' in Q:
        alpha = Q['alpha']
    else:
        alpha = 0
    if 'beta' in Q:
        beta = Q['beta']
    else:
        beta = 0
    if 'delta' in Q:
        delta = Q['delta']
    else:
        delta = 0
    if 'gamma' in Q:
        gamma = Q['gamma']
    else:
        gamma = 0
        
    A = np.array([[np.exp( 1j*(delta + alpha + gamma) )*np.cos(beta),
                   np.exp( 1j*(delta + alpha - gamma) )*np.sin(beta)],
                  [-np.exp( 1j*(delta - alpha + gamma) )*np.sin(beta),
                   np.exp( 1j*(delta - alpha - gamma) )*np.cos(beta)]])
    
    return A