# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:25:16 2019

@author: Lucas
"""

import numpy as np
import LowLevelQCAlgorithms as llqca
from pyquil.gates import CNOT, CCNOT, X, H
from pyquil.quil import Program

def phaseEstimation(U, a, t):
    
    p = Program()
    p.inst( initializePEA(U, a, t) )
    p.inst( qft(a) )
    


def initializePEA(U, a, c, t):
    """
    Return a program which transforms a state |v> into the state
    |v>|0> + U|v>|1> + U^2|v>|2> + ... + U^n|v>|n>, where U is some unitary
    operation, and |n> is the qubit basis state in the auxiliary qubitswhich 
    is the binary representation of n (e.g. |5> = |101>). 
    
    This method is the first step in the phase estimation algorithm, whose 
    purpose is to find the phase imparted by a unitary operator. The method 
    works by first making an equal superposition out of the auxiliary qubits 
    (those which are used to find the phase) -- that is, it turns the state
    |v> into (1/sqrt(n))|v>(|0> + |1> + ... + |n>). Then, using another 
    auxiliary qubit (the control qubit) U is applied to the nth basis state
    n times. This is done with a For loop where, during the ith iteration, the
    the control qubit is negated on the ith basis state and U (controlled by
    the control qubit) is applied. This puts the computer in the state 
    specified above.
    
    Parameters
    ----------
    U: pyquil Program
        This program is an implementation of the unitary operation U.
    a: list
        Holds the qubit positions of the auxiliary qubits which are used to 
        measure the phase of U.
    c: list 
        Holds the qubit position of the qubit which controls the application 
        of U
    t: list
        Holds the qubit location of the temporary qubit used for the 
        k-controlled And operation.
        
    Returns
    -------
    p: pyquil Program
        Program which puts the computer into the superposition described
        above.
    """

    n_qubits = len(a)
    M = 2**n_qubits
    p = Program()

    # Create equal superposition
    for i in a:
        p.inst(Program(H(i)))
    
    # Flip qubit which controls U
    p.inst( Program(X(c[0])) )
    
    # Apply phase gate to the ith basis vector i times
    for i in range(M):
        # Get control string for current basis vector
        ctrl_str = np.binary_repr(i, width=n_qubits)
        ctrl_str = [int(j) for j in ctrl_str]
        ctrl_str.reverse()

        # Negate control qubit in ith basis vector
        for j in range(len(ctrl_str)):
            if not ctrl_str[j]:
                p.inst(X(a[j]))
                
        p.inst( llqca.And(a, c, t) )
        
        for j in range(len(ctrl_str)):
            if not ctrl_str[j]:
                p.inst( X(a[j]) )

        # Apply gate controlled by auxiliary qubit
        p.inst(U)
        
    return p

def genNQubitTransform(U, n_qubits, t):
    """
    Return pyQuil Program which implements the n-qubit tranformation
    represented by the unitary matrix `U` (in standard ordering).
    
    This algorithm is based on the one presented by Rieffel and Polak in their
    book, "Quantum Computing: A Gentle Introduction". The algorithm is given
    on pages 89-91 in section 5.4.4. The basic idea is to take a unitary
    matrix and try to "undo" its action on a particular basis state with a
    k-controlled single qubit transformation Wm. The result of applying U and 
    Wm just leaves that basis state untouched. If you continue this process, 
    finding W(m-1),... then eventually UWmW(m-1)...W0 = I. Taking the inverse
    of each Wm (the inverses are labeled Cm) and multiplying them all by the
    identity just gives you back U. But since each Wm is a k-controlled single
    qubit transformation, so too is each Cm.
    
    Parameters
    ----------
    U: ndarray, dtype=np.complex_
        Complex-valued unitary matrix which will be broken down into
        k-controlled single-qubit gates.
    n_qubits: integer
        Number of qubits in your register. U should be of size
        2**n_qubits x 2**n_qubits.
    t: list
        List of 2 length-1 lists which hold the temporary qubit locations.
        
    Returns
    -------
    p: pyQuil Program
        Program consisting of single-qubit gates and CCNOT gates which
        perform the n_qubit tranformation defined by `U`.
    """
    
    if U.shape[0] is not U.shape[1]:
        raise Exception("Matrix is not square")
    if U.shape[0] is not 2**n_qubits:
        raise Exception("Matrix does not have appropriate size")
        
    # Put U in gray code ordering
    P = permutationMatrix(2**n_qubits)
    gray_code_U = P@U@P.T
    
    # Iteratively enerate Cm tranforms which will generate U
    p = Program()
    for m in range(2**n_qubits):
        print(m)
        v = gray_code_U[:, m]
        p = genCm(v, n_qubits, t).inst(p)
        Wm = genWmMat(v, n_qubits)
        gray_code_U = Wm@gray_code_U
        
    return p

def genCm(v, n_qubits, t):
    """
    Return program which transforms the first nonzero basis vector of the 
    input vector into the state represented by the input vector.
    
    This is part of the algorithm to create an arbitrary n-qubit
    transformation. It functions by iteratively adding in the phase, and
    transferring the amplitude of the highest nonzero basis state to the next
    highest basis state until the state is the same as that represented by the
    input vector `v`.
    
    Parameters
    ----------
    v: ndarray, dtype=np.complex_
        A complex vector of length 2**n_qubits which represents the state to
        be transformed into.
    n_qubits: integer
        Represents the number of qubits.
    t: list
        List of 2 length-1 lists which hold the temporary qubit locations.
        
    Returns
    -------
    p: pyQuil Program
        Program which tranforms the first nonzero basis state of `v` into 
        state v.
    """
    
    trimmed_v, nonzero_places = parseVector(v, n_qubits)
    gc_basis = getGraycodeBasis(n_qubits, nonzero_places)
    theta, phi = calcAmplitudesPhases(trimmed_v)
    
    p = Program()
    
    # Change global phase so that last basis vector has no phase
    a, b, z = getControlStrings(gc_basis[-2], gc_basis[-1], n_qubits)
    Q = {'delta': phi[-1]/2, 'alpha': -phi[-1]/2}
    p.inst( llqca.kControlGate(z, Q, a, b, t) )
    
    # For basis vectors 0 to n - 1, add amplitude and phase to make vector
    for i in range(np.size(theta), nonzero_places[0], -1):
        
        a, b, z = getControlStrings(gc_basis[i - 1], gc_basis[i], n_qubits)
        
        # Add phase and amplitude to next basis vector
        Q1 = {'delta': phi[i - 1]/2, 'alpha': phi[i - 1]/2}
        Q2 = {'beta': -theta[i - 1]}
        p = llqca.kControlGate(z, Q1, a, b, t).inst(p)
        p = llqca.kControlGate(z, Q2, a, b, t).inst(p)
        
    return p


def genWmMat(v, n_qubits):
    """
    Return matrix (in gray code ordering) which transforms the input state 
    into the first nonzero basis vector (in gray code ordering) of the vector.
    
    This is part of the algorithm to create an arbitrary n-qubit
    transformation. It functions by iteratively canceling out the phase, and
    transferring the amplitude of the highest nonzero basis state to the next
    highest basis state until the entire amplitude of the state is in the 
    lowest nonzero basis vector. 
    
    Parameters
    ----------
    v: ndarray, dtype=np.complex_
        A complex vector of length 2**n_qubits which represents the state to
        be transformed.
    n_qubits: integer
        Represents the number of qubits.
        
    Returns
    -------
    Wm: ndarray, dtype=np.complex_
        Complex matrix which tranforms v into the mth basis state. 
    """
    
    trimmed_v, nonzero_places = parseVector(v, n_qubits)
    theta, phi = calcAmplitudesPhases(trimmed_v)
    
    # Generate W_m in matrix form
    Wm = np.eye( 2**n_qubits, dtype=np.complex_ )
    Wm[ -1, -1] =  np.exp(-1j*phi[-1])
    
    # For basis vectors n - 1 to 0, cancel phase, transfer amplitude to 
    # next basis vector down
    for i in range(np.size(theta), nonzero_places[0], -1):
    
        # Generate matrix transformations corresponding to the control gates
        Q1_mat = np.eye(2**n_qubits, dtype=np.complex_)
        Q2_mat = np.eye(2**n_qubits, dtype=np.complex_)
        Q1_mat[i - 1, i - 1] =  np.exp(-1j*phi[i - 1])
        Q2_mat[i - 1:i + 1, i - 1:i + 1] = np.array(
                [[np.cos(theta[i - 1]), np.sin(theta[i - 1])],
                 [-np.sin(theta[i - 1]), np.cos(theta[i - 1])]] 
                )
    
        # Add this iteration to the matrix representation of W_m
        Wm = Q2_mat@Q1_mat@Wm
        
    return Wm

def genWm(v, n_qubits, t):
    """
    Return program which transforms the input state into the first nonzero 
    basis vector (in gray code ordering) of the vector.
    
    This is part of the algorithm to create an arbitrary n-qubit
    transformation. It functions by iteratively canceling out the phase, and
    transferring the amplitude of the highest nonzero basis state to the next
    highest basis state until the entire amplitude of the state is in the 
    lowest nonzero basis vector. 
    
    Parameters
    ----------
    v: ndarray, dtype=np.complex_
        A complex vector of length 2**n_qubits which represents the state to
        be transformed.
    n_qubits: integer
        Represents the number of qubits.
    t: list
        List of 2 length-1 lists which hold the temporary qubit locations.
        
    Returns
    -------
    p: pyQuil Program
        Program which tranforms the state v into the mth basis state (in gray
        code ordering).
    """
    
    trimmed_v, nonzero_places = parseVector(v, n_qubits)
    gc_basis = getGraycodeBasis(n_qubits, nonzero_places)
    theta, phi = calcAmplitudesPhases(trimmed_v)
    
    # Change global phase so that last basis vector has zero phase
    a, b, z = getControlStrings(gc_basis[-2], gc_basis[-1], n_qubits)
    Q = {'delta': -phi[-1]/2, 'alpha': phi[-1]/2}
    p = Program()
    p.inst( llqca.kControlGate(z, Q, a, b, t) )
    
    # For basis vectors n - 1 to 0, cancel phase, transfer amplitude to 
    # next basis vector down
    for i in range(np.size(theta), nonzero_places[0], -1):
        
        a, b, z = getControlStrings(gc_basis[i - 1], gc_basis[i], n_qubits)
        
        # Get rid of phase and amplitude of next basis vector
        Q1 = {'delta': -phi[i - 1]/2, 'alpha': -phi[i - 1]/2}
        Q2 = {'beta': theta[i - 1]}
        p.inst( llqca.kControlGate(z, Q1, a, b, t) )
        p.inst( llqca.kControlGate(z, Q2, a, b, t) )
        
    return p

def getGraycodeBasis(n_qubits, nonzero_places):
    """
    Return list of digits in graycode ordering with the end trimmed to match
    the trimmed vector.
    
    Parameters
    ----------
    n_qubits: integer
        Number of qubits. Basis length will be 2**n_qubits.
    nonzero_places: ndarray
        List of nonzero places in an original vector of length 2**n_qubits.
        The graycode list will be trimmed according to the last nonzero place
        in this vector.
        
    Returns
    -------
    trimmed_gc_basis: List
        List of digits representing a basis of binary strings in graycode
        ordering. The end is trimmed off to correspond with the trimming done
        to the original vector.
    """
    
    std_basis = [i for i in range(2**n_qubits)]
    P = permutationMatrix(2**n_qubits)
    gc_basis = P@std_basis
    trimmed_gc_basis = gc_basis[:(nonzero_places[-1] + 1)]
    
    return trimmed_gc_basis

def parseVector(v, n_qubits):
    """
    Tests to make sure input vector has appropriate length and not too many
    zeros, then returns vector with trailing zeros trimmed off. Includes list 
    of nonzero places in original vector. 
    
    Parameters
    ----------
    v: ndarray, dtype=np.complex_
        Vector of length 2**n_qubits which should be trimmed.
    n_qubits: integer
        Number of qubits.
        
    Returns
    -------
    trimmed_v: ndarray, dtype=np.complex_
        Original vector v with leading and following zeros trimmed off.
    nonzero_places: ndarray
        List of index locations in original vector where nonzer entries
        resided.
    """
    threshold = 1e-13
    nonzero_places = (np.absolute(v) > threshold).nonzero()[0]
    
    if len(v) is not 2**n_qubits:
        raise Exception("vector does not have proper length")
    if nonzero_places is 0:
        raise Exception("vector must have at least one nonzero component")
        
    trimmed_v = v[:(nonzero_places[-1] + 1)]
    
    return trimmed_v, nonzero_places

def calcAmplitudesPhases(v):
    """
    Returns phases associated with each basis vector, and rotation angles
    needed to transform vector into the lowest nonzero basis state.
    
    Part of the larger algorithm which creates arbitrary n-qubit
    transformations using 1- and 2-qubit gates.
    
    Parameters
    ----------
    v: ndarray, dtype=np.complex_
        Complex-valued vector of length n.
        
    Returns
    -------
    theta: ndarray
        Real-valued vector of length (n - 1) which represents rotation angles
        which, when applied to v, transform it into the lowest nonzero basis
        state.
    phi: ndarray
        Real-valued vector of length n which represents the complex phases
        associated with each coordinate of v.
    """
    
    a = np.absolute(v)
    c = np.zeros( np.size(v) )
    c[-1] = a[-1]
    for i in range(np.size(c) - 1, 0, -1):
        c[i - 1] = np.sqrt(a[i - 1]**2 + c[i]**2)
        
    theta = np.arcsin(c[1:]/c[0:-1])
    phi = np.angle(v)
    
    return theta, phi

def permutationMatrix(n):
    """
    Return a permutation matrix which, when multiplied on the right of a list
    in the standard ordering, produces a list in the gray code ordering.
    
    This works by making a list in the standard ordering 
    ([0, 1, ..., n]), and then uses bitshifts to put these numbers
    in the standard gray code ordering. Then a permutation matrix is created
    using these two lists.
    
    Parameters
    ----------
    n: integer
        Length of basis state list. 
        
    Returns
    -------
    P: ndarray
        Permutation matrix which takes vectors in standard basis and
        reorders them into vectors in the graycode basis. Dimensions are
        n x n.
    """
    
    standard = [i for i in range(n)]
    graycode = np.bitwise_xor( standard, np.right_shift( standard , 1) )
    
    P = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        P[i, graycode[i]] = 1
        
    return P

def getControlStrings(zero_string, one_string, n_qubits):
    """
    Return list of indices representing the control register, the target
    register, and the binary control string (with the bit denoting Q vs. XQX),
    given the "zero string" and the "one string". 
    
    Here the "zero string" is the string representing the basis vector which
    the gate will treat as 0, and the "one string" it will treat as 1. All 
    other basis vectors will remain unchanged. The "zero string" and the
    "one string" should differ by exactly one qubit, and that qubit will be
    the target qubit. If the "zero string" has a 0 in the target qubit place,
    a k-controlled gate will be applied. If it has a 1 in the target qubit
    place, a k-controlled gate XQX will be applied.
    
    Parameters
    ----------
    zero_string: integer
        An integer whose binary representation is the "zero string".
    one_string: integer
        An integer whose binary representation is the "one string".
            
    Returns
    -------
    a: list
        Length-k list holding the locations of the control register.
    b: list
        Length-1 list holding the location of the target qubit.
    z: list
        Length-2 list of lists, where the first sublist is length-k and holds
        the control bitstring pattern, and the second sublist is length-1 and
        holds the extra control bit.
    """
    
    # Make logic array with 1 in the place of the target qubit
    is_target_bitstring = np.bitwise_xor(zero_string, one_string)
    is_target_qubit = [int(i) for i in 
                       np.binary_repr(is_target_bitstring, width=n_qubits)]
    
    if sum(is_target_qubit) is not 1:
        raise Exception(
                "zero string and one string differ in more than one place")
    
    # Bitstring and list orderings are reversed
    is_target_qubit.reverse()

    # Find which qubits are target and control
    b = [i for i in range(len(is_target_qubit)) if is_target_qubit[i] == 1]
    a = [i for i in range(len(is_target_qubit)) if is_target_qubit[i] == 0]
    
    # Get binary string control sequence
    control_seq = [int(i) for i in 
                   np.binary_repr(zero_string, width=n_qubits)]
    control_seq.reverse()
    z = [[], []]
    z[0] = [control_seq[i] for i in a]
    
    # Determine which state target qubit is in for first basis state
    z[1] = [control_seq[i] for i in b]
    
    return a, b, z