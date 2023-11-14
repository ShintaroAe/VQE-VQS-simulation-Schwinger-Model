from qiskit.circuit import ParameterVector, Parameter
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, execute, Aer
from qiskit.algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from qiskit.primitives import BackendEstimator
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
from scipy.optimize import minimize
from numpy.random import default_rng
from qiskit.providers.fake_provider import FakeAthensV2
from qiskit.tools.visualization import plot_histogram
import time

plt.rcParams['font.family']="arial unicode ms"
plt.rcParams['font.size']=14

pi = np.pi

class ExactDiagonalization:
    def __init__(self, num_qubits, theta, J, m, w):
        self.num_qubits = num_qubits
        self.theta = theta # initial theta
        self.J = J
        self.m = m
        self.w = w

    def hamiltonian(self, theta):
        N = self.num_qubits
        basis = spin_basis_1d(N)

        coef_ZZ = [[self.J/2*(N-1-i), i, j] for i in range(1, N-1) for j in range(i)]
        coef_Z = []
        for i in range(N):
            temp = 0
            temp += self.J/2*((N-i-(i%2))//2)
            temp += self.m/2*(-1)**i
            temp += theta*self.J/2/pi*(N-1-i)
            coef_Z.append([temp, i])
        coef_XXYY = [[self.w/2, i, i+1] for i in range(N-1)]

        static = [["zz", coef_ZZ], ["z", coef_Z], ["xx", coef_XXYY], ["yy", coef_XXYY]]
        dynamic = []
        H = hamiltonian(static, dynamic, basis = basis, check_herm=False, check_pcon=False, check_symm=False)
        return H

    def initialize(self, sample=False):
        H = self.hamiltonian(self.theta)
        energy, state = H.eigsh(k = 1, which="SA")
        self.init_energy = energy
        self.init_state = state

    def chiral_condensate(self, del_theta, duration, steps):
        H = self.hamiltonian(self.theta+del_theta)
        N = self.num_qubits
        basis = spin_basis_1d(N)
        self.initialize()
        init_state = self.init_state.reshape((-1))
        time_list = np.linspace(0, duration, steps)
        states = H.evolve(init_state, 0, time_list)

        self.time_list  =time_list
        self.states = states

        coef_cc = [[(-1)**n*(self.J/self.w)**0.5/N, n] for n in range(N)]
        static_cc = [['z', coef_cc]]
        dynamic = []
        H_cc = hamiltonian(static_cc, dynamic, basis = basis, check_herm=False, check_pcon=False, check_symm=False)
        obs_list = obs_vs_time(states, time_list,dict(cc=H_cc))["cc"]

        return obs_list, time_list
    
    def ratefunction(self, del_theta, start, end, steps):
        '''
        return the rate function and time list.
        input:
            duration: float
            steps: int
        output: (np.array, np.array)
            (rate function list, time list)
        '''
        H = self.hamiltonian(self.theta+del_theta)
        self.initialize()
        init_state = self.init_state.reshape((-1))
        time_list = np.linspace(start, end, steps)
        states = H.evolve(init_state, 0, time_list)

        self.time_list  =time_list
        self.states = states

        ratefunction = -np.log(abs(np.inner(np.conj(init_state), states.T)))/self.num_qubits # type: ignore

        return ratefunction, time_list
    
def compare_fidelity(LE, del_theta):
    '''
    function that returns the list of fidelity and the time_list between LoschmidtEcho class and exact diagonalization. 
    input:
        LE: LoschmidtEcho
    output: (np.array, np.array)
        np.array of fidelitiy
        np.array of time
    '''
    n = len(LE.time_list)
    fid_list = []
    edd = ExactDiagonalization(LE.num_qubits, LE.theta, LE.J, LE.m, LE.w)
    _, __ = edd.ratefunction(del_theta, start=LE.time_list[0], end=LE.time_list[-1], steps=len(LE.time_list))
    states = edd.states
    states_ed = np.zeros([states.shape[1], states.shape[0]], dtype=np.complex128)
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            states_ed[j][-i-1] = states[i][j]
    for i in range(n):
        state_ed = Statevector(states_ed[i])
        qc = LE.ansatz.bind_parameters({LE.params: LE.params_list[i]})
        state_vqe = Statevector(qc)
        fid_list.append(state_fidelity(state_ed, state_vqe))
    return fid_list, LE.time_list