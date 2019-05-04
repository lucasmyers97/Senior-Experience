# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 17:29:09 2019

@author: Lucas
"""

from math import floor
from pyquil.quil import Program
from pyquil.gates import CNOT, CCNOT, X, RY, RZ, PHASE
from ForestSpecificMethods import *

def Flip(a, b):
    """
    Return pyQuil program which negates qubit b_i exactly when the 
    conjunction of qubits a_0 through a_(i + 1) is true.
    
    Given two disjoint lists of integers `a` of length m and `b` of length 
    m - 1 representing qubit locations, return a program which negates qubit 
    b_i exactly when qubits a_0 through a_(i + 1) are true.
    
    See Rieffel and Polak pg. 115.
    
    Parameters
    ----------
    a : list
        Holds the locations of the qubits in length-m register a. `a` must be 
        length-m.
    b : list
        Holds the locations of the qubits in length-(m - 1) register b.
        `b` must be of length-(m - 1), and disjoint from `a`.
        
    Returns
    -------
    p : pyQuil program
        Set of instructions which carries out the Flip operation
        
    """
    
    m = len(a)
    
    if len(b) is not m - 1:
        raise Exception("b does not have appropriate length")
        
    if not set(a).isdisjoint(b):
        raise Exception("a and b are not disjoint")
        
    p = Program()
    if m is 2:
        p.inst( CCNOT(a[1], a[0], b[0]) )
    else:
        p.inst( CCNOT(a[m - 1], b[m - 3], b[m - 2]) )
        p.inst( Flip(a[0:(m - 2) + 1], b[0:(m - 3) + 1]) )
        p.inst( CCNOT(a[m - 1], b[m - 3], b[m - 2]) )
        
    return p
    
def AndTemp(a, b, c):
    """
    Return pyQuil program which places the conjunction of the bits in register
    a in the single-qubit register b, making temporary use of the qubits 
    in register c.
    
    Given three disjoint lists of integers, `a` of length m, `b` of length 1,
    and `c` of length m - 2 representing qubit locations, return a program
    which negates the qubit in register `b` exactly when the conjunction of 
    all qubits in register `a` is true.
    
    See Rieffel and Polak pg. 115.
    
    Parameters
    ----------
    a: list
        Holds the locations of qubits in length-m register a. `a` must be 
        length-m.
    b: list
        Holds the locations of qubits in length-1 register b. `b` must be
        length-1 and disjoint from `a`.
    c: list
        Holds the locations of qubits in length-(m - 2) register c. `c` must 
        be length-(m - 2) and disjoint from `c`.
        
    Returns
    -------
    p : pyQuil program
        Set of instructions which carries out the AndTemp operation
    """
    
    m = len(a)
    
    if len(b) is not 1:
        raise Exception("b does not have appropriate length")
    if len(c) is not m - 2:
        raise Exception("c does not have appropriate length")
        
    if not set(a).isdisjoint(b):
        raise Exception("a and b are not disjoint")
    if not set(a).isdisjoint(c):
        raise Exception("a and c are not disjoint")
    if not set(b).isdisjoint(c):
            raise Exception("b and c are not disjoint")
        
    p = Program()
    if m is 2:
        p.inst( CCNOT(a[1], a[0], b[0]) )
    else:
        p.inst( Flip(a, (c + b)) )
        p.inst( Flip(a[0:(m - 2) + 1], c) )
        
    return p
    
def And(a, b, t=[]):
    """
    Return program that flips the qubit in the length-1 register b if and only
    if all quibits in a are 1. 
    
    Given two disjoint lists of integers, `a` of length-m and `b` of length-1
    with each representing qubit locations, return a pyQuil program which 
    negates the qubit in register b exactly when all of the qubits in register
    a are 1. This uses a single temporary qubit.
    
    See Rieffel and Polak pg. 116 for an explanation of this algorithm. Note
    that (at least in the 2011 version), there is an error in the definition 
    and use of j in the book. I have changed the definition of j, and the
    indices of several of the registers therein. I believe that my version
    works.
    
    Parameters
    ----------
    a: list
        Holds the locations of qubits in length-m register a. `a` must be
        length-m.
    b: list
        Holds the locations of qubits in length-1 register b. `b` must be
        length-1, and must be disjoint from `a`. 
    t: list
        Holds the locations of qubits in length-1 temporary register t. `t`
        must be length-1, and must be disjoint from both `a` and `b`.
        
    Returns
    -------
    p : pyQuil program
        Set of instructions which carries out the And operation
    """
    
    m = len(a)
    
    if len(b) is not 1:
        raise Exception("b does not have appropriate length")
    if not set(a).isdisjoint(b):
        raise Exception("a and b are not disjoint")
    if not set(a).isdisjoint(t):
        raise Exception("a and t are not disjoint")
    if not set(b).isdisjoint(t):
        raise Exception("b and t are not disjoint")
    
    if m is 1:
        return Program(CNOT(a[0], b[0]))
    elif m is 2:
        return Program(CCNOT(a[1], a[0], b[0]))
    else:
        if len(t) is not 1:
            raise Exception("t does not have appropriate length")
            
        p = Program()
        
        k = floor(m/2)
        if m % 2:
            l = 0
        else:
            l = 1
            
        j = k - 2 - l
            
        p.inst( AndTemp(a[k:(m - 1) + 1], t, a[0:j + 1]) )
        p.inst( AndTemp(a[0:(k - 1) + 1] + t, b, a[k:(k + j + l) + 1]) )
        p.inst( AndTemp(a[k:(m - 1) + 1], t, a[0:j + 1]) )
        
        return p
    
def Swap(a):
    """
    Return program that permutes the qubits as follows: qubit 0 becomes qubit
    k and qubits 1 through k become qubits 0 through (k - 1).
    
    Given a list of integers `a` representing qubit locations in a register a,
    permute the qubits so that qubit 0 becomes qubit k and qubits 1 through k
    become qubits 0 through (k - 1) (respectively). This is done by performing
    (k - 1) Swap operations (see page 79 in Rieffel and Polak) on adjacent
    qubits.
    
    Parameters
    ----------
    a: list
        Holds the locations of qubits in the length-k qubit register a. 
        
    Returns
    -------
    p: pyQuil Program
        Set of instructions which carries out the qubit permutation described
        above.
    """   
    
    p = Program()
    
    for i in range(len(a) - 1):
        p.inst(CNOT(a[i], a[i + 1]), 
               CNOT(a[i + 1], a[i]), 
               CNOT(a[i], a[i + 1]))
    
    return p

def arbitraryGate(Q, t):
    """
    Return pyQuil program which applie a gate defined by dictionary `Q` to 
    target qubit `t`.
    
    In Rieffel and Polak, they show that an arbitrary single-qubit
    transformation can be uniquely defined by four real numbers. This function
    uses the same convention, naming these numbers `alpha`, `beta`, `delta`, 
    and `gamma`, and using a dictionary `Q` to hold each number indexed by the 
    same name. 
    
    Parameters
    ----------
    Q: dictionary
        Holds the four angles `alpha`, `beta`, `delta`, and `gamma` used to
        specify a single-qubit transformation by the Rieffel and Polak
        convention. These angles are indexed by their names.
        
    t: list
        Length-1 list holding the qubit location of target qubit t.
        
    Returns
    -------
    p: pyquil Program
        Program which executes the arbitrary single-qubit transformation.
    """
    
    if len(t) is not 1:
        raise Exception("target qubit t must have length 1")
    
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
    
    p = Program( RZ(-2*gamma, t[0]), RY(-2*beta, t[0]),
                 RZ(-2*(alpha + delta), t[0]), PHASE(2*delta, t[0]) )
    
    return p

def controlGate(c, t, Q):
    """
    Returns program that applies a single-qubit control gate to a target 
    qubit.
    
    An arbitrary gate is specified, in the style of Rieffel and Polak, by four
    rotation angles: alpha, beta, delta, gamma. These real numbers uniquely
    specify a single qubit tranformation, and are stored in a dictionary `Q`.
    They are indexed by the names above. The control qubit `c` controls the
    application of the transformation, and the target qubit `t` is what the
    transformation is applied to.
    
    See Rieffel and Polak pgs. 86-87 for an explanation of this algorithm.
    
    Parameters
    ----------
    c: list
        Holds the location of the control qubit. This is a single qubit, so 
        the list must be length-1.
    t: list
        Holds the location of the target qubit. This is a single qubit, so the
        list must be length-1. Additionall, it must be a different qubit from 
        the control qubit. 
    Q: dictionary
        Holds the four real numbers (angles) necessary to specify an arbitrary
        single-qubit transformation. Indexed by the names of the angles, 
        namely: alpha, beta, delta, gamma. 
        
    Returns
    -------
    p: pyQuil Program
        Set of instructions which carries out a single-qubit control gate 
        transformation. 
    """
    
    if len(c) is not 1:
        raise Exception("control qubit list must be length-1")
    if len(t) is not 1:
        raise Exception("target qubit list must be length-1")
    if t[0] is c[0]:
        raise Exception("target and control qubits must be different")
    
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
    
    p = Program()
    
    p.inst( RZ(alpha - gamma, t[0]), CNOT(c[0], t[0]) )
    p.inst( RZ(alpha + gamma, t[0]), 
            RY(beta, t[0]), CNOT(c[0], t[0]) )
    p.inst( RY(-beta, t[0]), RZ(-2*alpha, t[0]) )
    p.inst( RZ(delta, c[0]), RZ(-delta, c[0]), PHASE(delta, c[0]) )
    
    return p

def kControlGate(z, Q, a, b, t):
    """
    Return program which applies a gate to target qubit b, if and only if 
    all of the bits in register a match the bit string z.
    
    Rieffel and Polak have a convention whereby they represent the two 
    different k-controlled qubit operators Q and XQX with the `and` symbol.
    Which of these is used is decided by an extra control bit in the control 
    string, applying Q if the extra bit is 0, and XQX otherwise. To 
    accommodate this, the list `z` is a list of lists, with the first list 
    being length-m and acting as the control string, and the second list being 
    length-1 and acting as the extra bit mentioned above. See Rieffel and 
    Polak pgs. 88-89 for an explanation of this convention.
    
    Additionally, the algorithm requires two temporary qubits, so t is a list 
    of lists, where each sublist holds the location of a temporary qubit.
    
    See Rieffel and Polak pgs. 116-117 for an explanation of this algorithm.
    
    Parameters
    ----------
    z: list
        Length-2 list of lists, where the first sublist is length-k and holds
        the control bitstring pattern, and the second sublist is length-1 and
        holds the extra control bit.
    Q: dictionary
        Holds the four real numbers (angles) necessary to specify an arbitrary
        single-qubit transformation. Indexed by the names of the angles, 
        namely: alpha, beta, delta, gamma. 
    a: list
        Length-k list. Holds the locations of the qubits in register a.
    b: list
        Length-1 list. Holds the location of the qubit in register b.
        Must be disjoint from the qubits in register a.
    t: list
        Length-2 list of lists. Each sublist is length-1. They hold the 
        locations of the first and second temporary qubits, respectively.
        
    Returns
    -------
    p: pyQuil Program
        Set of instructions that carries out a k-qubit control gate
        transformation.
    
    """
    
    k = len(z[0])
    
    if len(a) is not k:
        raise ValueError("a must be of length k")
    if len(z[1]) is not 1:
        raise ValueError("z[1] must have length 1")
    if len(b) is not 1:
        raise ValueError("target qubit register b must have length 1")
    if len(t[0]) is not 1:
        raise ValueError("First temporary qubit register must have length 1")
    if len(t[1]) is not 1:
        raise ValueError("Second temporary qubit register must have length 1")
        
    p = Program()    
        
    if z[1][0] == 1:
        p.inst(X(b[0]))
    
    for i in range( len(z[0]) ):
        if z[0][i] == 0:
            p.inst(X(a[i]))
        
    p.inst(And(a, t[0], t[1]))
    p.inst(controlGate(t[0], b, Q))
    p.inst(And(a, t[0], t[1]))
    
    for i in range( len(z[0]) ):
        if z[0][i] == 0:
            p.inst(X(a[i]))
            
    if z[1][0] == 1:
        p.inst(X(b[0]))
            
    return p