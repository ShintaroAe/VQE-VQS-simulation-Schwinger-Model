from qiskit.circuit import ParameterVector, Parameter
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, execute, Aer
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator
from qiskit.algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import BackendEstimator
# from quspin.operators import hamiltonian
# from quspin.basis import spin_basis_1d
from scipy.optimize import minimize
from numpy.random import default_rng
from qiskit.providers.fake_provider import FakeAthensV2
from qiskit.tools.visualization import plot_histogram
import time

plt.rcParams['font.family']="arial unicode ms"
plt.rcParams['font.size']=14

pi = np.pi
class LoschmidtEcho:    
    def __init__(self, num_qubits, theta, backend, J, m, w):
        self.num_qubits = num_qubits
        self.theta = theta
        self.ansatz_list = []
        '''ansatz_list is the list of tuple with four elements
        (name, place, function name, underlying pauli string)
        name:
            name of the ansatz
            e.g., 'z', 'odd_XXYY', 'even_XXYY', 'odd_ZZ', 'even_ZZ'
        place:
            index of each ansatz. starting from zero to "the number of that ansatz in one layer" - 1
            e.g.0, 1, 2, ...
        fun:
            corresponding conditional function
        sigma:
            sigma strings that determines the rotation
            e.g., "IIIIXX", "IIIZII", ...
        '''
        self.ansatz = QuantumCircuit(self.num_qubits)
        self.ansatz.x(i for i in range(self.num_qubits) if i%2==0)
        self.params = ParameterVector("params")
        self.backend = backend
        self.init_parameters = np.array([])
        self.params_list = np.array([])
        self.estimator_obs = SparsePauliOp(["I"*self.num_qubits + "Z"], coeffs=[1]) # pay attention to the order of qubits!
        self.J = J
        self.m = m
        self.w = w
        self.__qc = QuantumCircuit(0, 0)
    
    def hamiltonian(self, theta):
        '''
        setting the Hamiltonian of this system
        input:
            theta: float
                value of theta. either self.theta or self.theta + del_theta
        output: list, list
            [list of Hamiltonian in Pauli strings, eg. "IIXX" means 'I' for first and second qubit, 'X' for third and forth qubit], [list of coeffs]
            paulis, coeffs = self.hamiltonian(theta)
            len(paulis) == len(coeffs) --> True
        '''
        paulis = [] # length of each pauli string is the number of qubits
        coeffs = []
        N = self.num_qubits ## just adjusting to my note (eq. (1))
        for i in range(1, N-1):
            for j in range(i):
                pauli_string = ["I" for i in range(N)]
                pauli_string[i] = "Z"
                pauli_string[j] = "Z"
                paulis.append(''.join(pauli_string))
                coeffs.append(self.J/2*(N-1-i))
        for i in range(N):
            pauli_string = ["I" for i in range(N)]
            pauli_string[i] = "Z"
            paulis.append(''.join(pauli_string))
            temp = 0
            temp += self.J/2*((N-i-(i%2))//2)
            temp += self.m/2*(-1)**i
            temp += theta*self.J/2/pi*(N-1-i)
            coeffs.append(temp)
        
        for i in range(N-1):
            for paul in ["X", "Y"]:
                pauli_string = ["I" for i in range(N)]
                pauli_string[i] = paul
                pauli_string[i+1] = paul
                paulis.append(''.join(pauli_string))
            coeffs.append(self.w/2)
            coeffs.append(self.w/2)
        
        return paulis, coeffs
    
    def ansatz_RXCNOT(self):
        """put HEA """
        n = len(self.params)
        self.params.resize(n + 2*self.num_qubits)
        for i in range(self.num_qubits):
            self.ansatz.rx(self.params[n+i], i)
        for i in range(self.num_qubits):
            self.ansatz.cx(2*i, 2*i+1)

    def ansatz_Z(self):
        '''put Z gates to the ansatz'''
        n = len(self.params)
        self.params.resize(n + self.num_qubits)
        for i in range(self.num_qubits):
            self.ansatz.rz(self.params[n+i], i)
        for i in range(self.num_qubits):
            sigma = ["I"]*self.num_qubits
            sigma[i] = "Z"
            sigma = ''.join(sigma)
            self.ansatz_list.append(("Z", i, self.conditional_Z, [sigma]))

    def ansatz_ZZ(self):
        '''put ZZ gates to the ansatz'''
        n = len(self.params)
        self.params.resize(n + self.num_qubits - 1)
        for i in range(1, (self.num_qubits-1)//2+1):
            self.ansatz.cx(2*i-1, 2*i)
            self.ansatz.rz(self.params[n+i-1], 2*i)
            self.ansatz.cx(2*i-1, 2*i)
        for i in range(1, (self.num_qubits-1)//2+1):
            sigma = ['I']*self.num_qubits
            sigma[2*i-1] = "Z"
            sigma[2*i] = "Z"
            sigma = ''.join(sigma)
            self.ansatz_list.append(("odd_ZZ", i-1, self.conditional_ZZ, [sigma]))

        for i in range(self.num_qubits//2):
            self.ansatz.cx(2*i, 2*i+1)
            self.ansatz.rz(self.params[n+(self.num_qubits-1)//2+i], 2*i+1)
            self.ansatz.cx(2*i, 2*i+1)
        for i in range(self.num_qubits//2):
            sigma = ['I']*self.num_qubits
            sigma[2*i] = "Z"
            sigma[2*i+1] = "Z"
            sigma = ''.join(sigma)
            self.ansatz_list.append(("even_ZZ", i, self.conditional_ZZ, [sigma]))

    def ansatz_XXYY(self):
        '''put XX + YY gates to the ansatz'''
        n = len(self.params)
        self.params.resize(n + self.num_qubits - 1)
        for i in range(1, (self.num_qubits-1)//2+1):
            self.ansatz.cx(2*i-1, 2*i)
            self.ansatz.h(2*i-1)
            self.ansatz.cx(2*i-1, 2*i)
            self.ansatz.rz(self.params[n+i-1], 2*i-1)
            self.ansatz.rz(-1*self.params[n+i-1], 2*i)
            self.ansatz.cx(2*i-1, 2*i)
            self.ansatz.h(2*i-1)
            self.ansatz.cx(2*i-1, 2*i)
        for i in range(1, (self.num_qubits-1)//2+1):
            sigma_XX = ["I"]*self.num_qubits
            sigma_XX[2*i-1] = 'X'
            sigma_XX[2*i] = "X"
            sigma_XX = ''.join(sigma_XX)
            sigma_YY = ["I"]*self.num_qubits
            sigma_YY[2*i-1] = 'Y'
            sigma_YY[2*i] = "Y"
            sigma_YY = ''.join(sigma_YY)
            self.ansatz_list.append(("odd_XXYY", i-1, self.conditional_XXYY, [sigma_XX, sigma_YY]))

        for i in range(self.num_qubits//2):
            self.ansatz.cx(2*i, 2*i+1)
            self.ansatz.h(2*i)
            self.ansatz.cx(2*i, 2*i+1)
            self.ansatz.rz(self.params[n+(self.num_qubits-1)//2+i], 2*i)
            self.ansatz.rz(-1*self.params[n+(self.num_qubits-1)//2+i], 2*i+1)
            self.ansatz.cx(2*i, 2*i+1)
            self.ansatz.h(2*i)
            self.ansatz.cx(2*i, 2*i+1)
        for i in range(self.num_qubits//2):
            sigma_XX = ["I"]*self.num_qubits
            sigma_XX[2*i] = 'X'
            sigma_XX[2*i+1] = "X"
            sigma_XX = ''.join(sigma_XX)
            sigma_YY = ["I"]*self.num_qubits
            sigma_YY[2*i] = 'Y'
            sigma_YY[2*i+1] = "Y"
            sigma_YY = ''.join(sigma_YY)
            self.ansatz_list.append(("even_XXYY", i, self.conditional_XXYY, [sigma_XX, sigma_YY]))


    '''
    Ansatz を足すときの注意点
    ansatz_Z, conditional_Zみたいに二つ足してくれればOK.
    そのときの引数は、元からあるものを参考に。
    self.ansatz_listにも参考に足すこと。
    A_ij とかのdiagonal の処理。
    '''

    def ratefunction(self, shots=10000, statevector=False):
        '''
        This the list of rate function and the list of time. including the initial state's rate function.
        input:
            shots: int
                number of shots. used if statevector = False
            statevector: bool
                whether statevector is used to get rate function.
        output: (np.array, np.array)
            (list of ratefunction(shape: (num_steps+1,)), list of time(shape: (num_steps+1)))
        '''
        le = self.abs_loschmidtecho(shots=shots, statevector=statevector)
        rf = -(np.log(abs(le)))/self.num_qubits
        return rf, self.time_list

    def abs_loschmidtecho(self, shots, statevector):
        '''
        Returns the absolute value of the Loschmidt echo.
        input: shots and statevector, same as ratefunction
        output: np.array
            shape: (steps+1, ) absolute value of the Loschmidt echo(float)
        '''
        result = np.zeros(len(self.time_list)) # what I'll return
        for i in range(len(self.time_list)):
            self.__qc = QuantumCircuit(self.num_qubits)
            # U(t)
            qc = self.ansatz.bind_parameters({self.params: self.params_list[i]})
            self.__qc.append(qc, range(self.num_qubits))
            # U^\dag(0)
            qc = self.ansatz.bind_parameters({self.params: self.params_list[0]})
            self.__qc.append(qc.inverse(), range(self.num_qubits))
            if not statevector:
                self.__qc.measure_all()
                job = execute(self.__qc, self.backend, shots=shots)
                counts = job.result().get_counts()
                keys = list(counts.keys())
                prob = np.array(list(counts.values()))/shots
                probability = 0
                for j in range(len(keys)):
                    if keys[j] == '0'*self.num_qubits:
                        probability = prob[j]
                result[i] = probability**0.5
            else:
                result[i] = abs(Statevector(self.__qc)[0]) ## so easy la?
            print(f"{i}/{len(self.time_list)}" if i%100 == 0 else "", end=" " if i%100 == 0 else "")
        return result

    def chiral_condensate(self, shots=10000, statevector=False):
        '''
        compute transition of chiral condensate
        input:
            shots: int
                number of shots
            you need to have set self.params_list beforehand
        output: (np.array, np.array)
            (array of chiral condensate by time, time)
            chiral condensate and time.
            size is (n, ) and (n, )
        '''
        result = [] ## what I'll return
        for params in self.params_list:
            self.__qc = QuantumCircuit(self.num_qubits+1)
            self.__qc.x(k for k in range(1, self.num_qubits+1) if k%2==1)
            coef = (self.J/self.w)**0.5/self.num_qubits
            for k in range(len(params)):
                name, place, fun, sigma = self.ansatz_list[k]
                fun(place, params[k], name, sigma)
            if statevector:
                est = Estimator()
                obs_list = []
                coef_list = []
                for i in range(self.num_qubits):
                    obs = ["I"] + ["I" for i in range(self.num_qubits)]
                    obs[i + 1] = "Z"
                    obs = "".join(obs)
                    obs = obs[::-1]
                    obs_list.append(obs)
                    coef_list.append(coef if i%2==0 else -coef)
                obs = SparsePauliOp(obs_list, coeffs = coef_list)
                job = est.run(self.__qc, obs)
                res = job.result()
                result.append(res.values[0])

            else:
                self.__qc.measure_all()
                job = execute(self.__qc, self.backend, shots=shots)
                counts = job.result().get_counts()
                keys = list(counts.keys())
                prob = np.array(list(counts.values()))/shots
                for k in range(len(keys)):
                    keys[k] = [*(keys[k][::-1])]
                    for l in range(len(keys[k])):
                        keys[k][l] = int(keys[k][l])
                exp = []
                for k in range(len(keys)):
                    key = keys[k]
                    temp = 0
                    for n in range(1, len(key)):
                        temp += ((-1)**(n-1))*(1-2*key[n])
                    exp.append(temp*coef)
                exp = np.array(exp)
                result.append(np.sum(exp*prob))
        result = np.array(result)
        return result, self.time_list

    def compute_energy(self, params, shots, statevector):
        '''
        return energy expectation of the Hamiltonian without constant term
        input: shots
        output: energy 
        '''
        if statevector:
            qc = self.ansatz.bind_parameters({self.params: params})
            pauli, coefs = self.hamiltonian(self.theta)
            for i in range(len(pauli)):
                paul = pauli[i]
                paul = paul[::-1]
                pauli[i] = paul
            obs = SparsePauliOp(pauli, coeffs=coefs)
            est = Estimator()
            job = est.run(qc, obs)
            result = job.result()
            return result.values[0]

        ## only calculating the energy without consant term
        qc = self.ansatz.bind_parameters({self.params: params})
        qc.measure_all()
        job = execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts()
        keys = list(counts.keys())
        prob = np.array(list(counts.values()))/shots
        for i in range(len(keys)):
            keys[i] = [*(keys[i][::-1])]
            for j in range(len(keys[i])):
                keys[i][j] = int(keys[i][j])

        ## giving the normal order of qubits and counts lists
        ## variables:
        ##      energy: what I'll return
        ##      energy_z: list of energy without prob
        ##      energy_zz: list of energy without prob
        ##      energy_xx: list of energy XX without prob
        ##      energy_yy: list of energy YY without prob
        ##      energy = (energy_Z + energy_XX + energy_YY) * for each probability
        energy = 0
        energy_z = []
        energy_zz = []
        for each_measurement in range(len(keys)):
            key = keys[each_measurement]
            ## calculation of zz part
            temp = 0
            for i in range(1, self.num_qubits-1):
                for j in range(i):
                    temp += self.J/2*(self.num_qubits-1-i)*(1-2*key[i])*(1-2*key[j])
            energy_zz.append(temp)
            ## calculation of z part
            temp = 0
            for i in range(self.num_qubits):
                coef = 0
                coef += self.J/2*((self.num_qubits-i-(i%2))//2)
                coef += self.m/2*(-1)**i
                coef += self.theta*self.J/2/pi*(self.num_qubits-1-i)
                temp += coef*(1-2*key[i])
            energy_z.append(temp)
        energy_zz = np.array(energy_zz)
        energy_z = np.array(energy_z)
        energy += np.sum(energy_zz*prob+energy_z*prob)

        ## in the measurement of (w/2)sum(xx+yy), the method on Honda et al.'s appendix can be utilized. However, I think there's no need to use it, as it only complicate the circuit.
        ## so, first, X basis is used. 
        qc = self.ansatz.bind_parameters({self.params: params})
        qc.h(qc.qubits)
        qc.measure_all()
        job = execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts()
        keys = list(counts.keys())
        prob = np.array(list(counts.values()))/shots
        for i in range(len(keys)):
            keys[i] = [*(keys[i][::-1])]
            for j in range(len(keys[i])):
                keys[i][j] = int(keys[i][j])
        energy_xx = []
        for each_measurement in range(len(keys)):
            key = keys[each_measurement]
            ## calculation of xx part
            temp = 0
            for n in range(self.num_qubits-1):
                temp += (1-2*key[n])*(1-2*key[n+1])*self.w/2
            energy_xx.append(temp)
        energy_xx = np.array(energy_xx)
        energy += np.sum(energy_xx*prob)


        ## lastly, ansatz is measured in y basis
        qc = self.ansatz.bind_parameters({self.params: params})
        qc.sdg(qc.qubits)
        qc.h(qc.qubits)
        qc.measure_all()
        job = execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts()
        keys = list(counts.keys())
        prob = np.array(list(counts.values()))/shots
        for i in range(len(keys)):
            keys[i] = [*(keys[i][::-1])]
            for j in range(len(keys[i])):
                keys[i][j] = int(keys[i][j])
        energy_yy = []
        for each_measurement in range(len(keys)):
            key = keys[each_measurement]
            ## calculation of xx part
            temp = 0
            for n in range(self.num_qubits-1):
                temp += (1-2*key[n])*(1-2*key[n+1])*self.w/2
            energy_yy.append(temp)
        energy_yy = np.array(energy_yy)
        energy += np.sum(energy_yy*prob)
        return energy

    def get_cost_val_VQE(self, params, shots, statevector):
        '''computing cost function for VQE
        input:
            - params: parameter
            - cost_val_list: list of values of cost function(global)
        output:
            - cost_val: value of cost function
        '''
        cost_val = self.compute_energy(params, shots, statevector)
        self.cost_val_list.append(cost_val)
        return cost_val

    def initialize_VQE(self, method = "COBYLA", maxiter=500, tol=0.01, shots=10000, statevector=False):
        '''
        Run the VQE and find the best parameter
        input:
            method:
                one of the following
                ‘Nelder-Mead’ (see here)
                ‘Powell’ (see here)
                ‘CG’ (see here)
                ‘BFGS’ (see here)
                ‘Newton-CG’ (see here)
                ‘L-BFGS-B’ (see here)
                ‘TNC’ (see here)
                ‘COBYLA’ (see here)
                ‘SLSQP’ (see here)
                ‘trust-constr’(see here)
                ‘dogleg’ (see here)
                ‘trust-ncg’ (see here)
                ‘trust-exact’ (see here)
                ‘trust-krylov’ (see here)
            maxiter: int
                Maximum number of function evaluations.
            tol: float
                Final accuracy in the optimization (not precisely guaranteed).This is a lower bound on the size of the trust region.
        output:
            energy: the result(optimized energy)
            params_VQE: parameters for the result
            cost_val_list: transition of the optimization
        '''
        if method != "COBYLA": tol = None
        params_init_VQE = 2*pi*np.random.rand(len(self.params))
        self.cost_val_list = []
        res_VQE = minimize(self.get_cost_val_VQE, params_init_VQE, args=(shots, statevector), method=method, options={'maxiter':maxiter, 'tol':tol})
        self.init_parameters = res_VQE.x
        # return res_VQE.fun, res_VQE.x, np.array(self.cost_val_list)
    
    def quench(self, del_theta, duration, steps, shots = 10000, statevector = False):
        '''
        function that calculate the evolution of parameters by euler method
        input:
            init_params: np.array
                shape: (n,) initial parameters for self.params.
            duration: float
                how much time the calculation of evolution done
            steps: int
                how much steps is taken
            shots: int
                shots for each step

        output: np.array()
            shape: (steps+1, len(self.params))
            list of parameters "including initial state"
        '''

        self.list_det = []
        self.list_cond = []

        init_params = self.init_parameters
        n = len(self.params)
        result = np.zeros((steps+1, n))
        result[0] = init_params
        del_t = duration/steps
        for step in range(steps):
            ## getting result[step+1] from result[step]
            M_ij = np.zeros((n, n))
            V_i = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    M_ij[i][j] = self.AR_ij(result[step], i, j, shots, statevector)+self.N_ij(result[step], i, j, shots, statevector)
                V_i[i] = self.CI_i(result[step], del_theta, i, shots, statevector) + self.W_i(result[step], del_theta, i, shots, statevector)
                print(f"{i}/{n}", end=" ")
            self.list_det.append(np.linalg.det(M_ij))
            self.list_cond.append(np.linalg.cond(M_ij))
            dlambda_dt = np.linalg.inv(M_ij) @ V_i
            result[step+1] = result[step] + del_t * dlambda_dt
            print("step {} finished".format(step), end=' ' if step%5 != 4 else '\n')
        self.params_list = result
        self.time_list = np.linspace(0, duration, steps+1)
        # return list_det

    def AR_ij(self, params, i, j, shots, statevector):
        '''
        Function that return A_ij matrices
        input:
            params: np.array 
                length is len(self.params)
            i: index i
            j: index j
            shots: number of shots to determine
        output:
        '''
        if i==j: ########ミス!!!!!#########
            name = self.ansatz_list[i][0]
            if not "XXYY" in name:
                return 1/4
            else:
                place = self.ansatz_list[i][1]
                place_ind = place*2+1 if name=="even_XXYY" else place*2+2 if name=="odd_XXYY" else  exec("raise SyntaxError('unexpected name, ' + str(name))")
                self.__qc = QuantumCircuit(self.num_qubits+1, 2)
                self.__qc.x(k for k in range(1, self.num_qubits+1) if k%2==1) # type: ignore
                for k in range(i):
                    name, place, fun, sigma = self.ansatz_list[k]
                    fun(place, params[k], name, sigma)
                if statevector:
                    est = Estimator()
                    sigma = self.ansatz_list[i][3][0]
                    sigma = sigma.translate(str.maketrans({"X": "Z", "Y": "Z"}))
                    sigma_est_ij = SparsePauliOp([sigma[::-1]+"I"], coeffs=[1])
                    job = est.run(self.__qc, sigma_est_ij)
                    result = job.result()
                    experiment = result.values[0]
                else:                
                    self.__qc.measure(place_ind, 0)
                    self.__qc.measure(place_ind+1, 1)
                    experiment = self.exp__qc(shots=shots)
                return (2-2*experiment)/4


        if i > j: i, j = j, i ## now j > i
        
        _, __, ___, sigma_types_i = self.ansatz_list[i] 
        _, __, ___, sigma_types_j = self.ansatz_list[j]

        ar_ij = 0 # what I'll return

        for sigma_i in sigma_types_i:
            for sigma_j in sigma_types_j: ## taking the summation. refer to the eqn above.
                self.__qc = QuantumCircuit(1+self.num_qubits, 1)
                ## be careful! 0th qubit of this circuit is ancillary bit!
                self.__qc.h(0)
                self.__qc.x(k for k in range(self.num_qubits+1) if k%2!=0) # type: ignore
                for index in range(len(params)):
                    name, place, fun, _ = self.ansatz_list[index]
                    #either of conditional_z, conditional_zz or conditional_xxyy...
                    if index == i:
                        fun(place, params[index], name, sigma_i, put_XCX=True)

                    elif index == j:
                        fun(place, params[index], name, sigma_j, put_C=True)
                        break
                    else:
                        fun(place, params[index], name, "")

                self.__qc.h(0)
                if statevector:   
                    est = Estimator()             
                    job = est.run(self.__qc, self.estimator_obs)
                    result = job.result()
                    ar_ij += result.values[0]/4
                else:
                    self.__qc.measure(0, 0)
                    job = execute(self.__qc, self.backend, shots=shots)
                    counts = job.result().get_counts()
                    increment = (counts.get("0", 0)-counts.get("1", 0))/shots/4
                    ar_ij += increment
        return ar_ij
    
    def CI_i(self, params, del_theta, i, shots, statevector):
        '''
        a function to calculate CI_i
        input:
            params: list of parameters
            del_theta: difference of theta
            i: index
        output: value of CI_i
        '''
        paulis, coeffs = self.hamiltonian(self.theta+ del_theta)
        temp = 0
        for j in range(len(paulis)):
            temp += coeffs[j]/2*self.C_ij(params, i, paulis[j], shots, statevector)
        return temp
    
    def C_ij(self, params, i, sigma_j, shots, statevector):
        '''
        a function to calculate each part of CI_i. note that this is not directly calculating CI_ij in the equation above. rather, this is only calculating the real part (R(<0|...|0>)) without h_j/2
        '''
        _, __, ___, sigma_types_i = self.ansatz_list[i]

        c_ij = 0 ## what I'll return

        for sigma_i in sigma_types_i:## taking the summation. refer to the eqn above.
            self.__qc = QuantumCircuit(1+self.num_qubits, 1)
            ## be careful! 0th qubit of this circuit is ancillary bit!
            self.__qc.h(0)
            self.__qc.x(k for k in range(self.num_qubits+1) if k%2!=0) # type: ignore
            for index in range(len(params)):
                name, place, fun, _ = self.ansatz_list[index]
                #either of conditional_z, conditional_zz or conditional_xxyy...
                if index == i:
                    fun(place, params[index], name, sigma_i, put_XCX=True)
                else:
                    fun(place, params[index], name, "")
            ## here, I need to put controlled pauli_j
            for k in range(len(sigma_j)):
                if sigma_j[k]=='I': pass
                elif sigma_j[k]=='X': self.__qc.cx(0, k+1)
                elif sigma_j[k]=='Y': self.__qc.cy(0, k+1)
                elif sigma_j[k]=='Z': self.__qc.cz(0, k+1)
                else: raise SyntaxError("unexpected name")
            self.__qc.h(0)
            ## caution! a = h_j/2 isn't included here!
            if statevector:
                est = Estimator()
                job = est.run(self.__qc, self.estimator_obs)
                result = job.result()
                c_ij += result.values[0]
            else:
                self.__qc.measure(0, 0)
                job = execute(self.__qc, self.backend, shots=shots)
                counts = job.result().get_counts()
                increment = (counts.get("0", 0)-counts.get("1", 0))/shots
                c_ij += increment
        return c_ij

    def N_ij(self, params, i, j, shots, statevector):
        ## expectation value is taken over the following two sigma strings, sigma_i and sigma_j
        _, __, ___, sigma_i_list = self.ansatz_list[i]
        _, __, ___, sigma_j_list = self.ansatz_list[j]

        exp_ij_list = []
        ## here, I'll put ansatz until i-1 and j-1
        for index, sigma_list in [(i, sigma_i_list), (j, sigma_j_list)]:
            exp_ij = 0
            for sigma_ij in sigma_list:
                ## Measure only the sufficient number of qubits
                num_cbit = self.num_qubits - sigma_ij.count('I')
                iteration = iter(range(num_cbit))
                ## ancilla bit is in __qc just for convention... only 1 --> self.num_qubits sites are used 
                self.__qc = QuantumCircuit(self.num_qubits+1, num_cbit)
                self.__qc.x(k for k in range(1, self.num_qubits+1) if k%2==1) # type: ignore

                for k in range(index):
                    name, place, fun, sigma = self.ansatz_list[k]
                    fun(place, params[k], name, sigma)

                if statevector:
                    est = Estimator()
                    sigma_est_ij = SparsePauliOp([sigma_ij[::-1]+"I"], coeffs=[1])
                    job = est.run(self.__qc, sigma_est_ij)
                    result = job.result()
                    exp_ij += result.values[0]

                else:
                    for k in range(self.num_qubits):
                        if sigma_ij[k] == "I": pass
                        elif sigma_ij[k] == "X":
                            self.__qc.h(k+1)
                            self.__qc.measure(k+1, next(iteration))
                        elif sigma_ij[k] == "Y":
                            self.__qc.sdg(k+1)
                            self.__qc.h(k+1)
                            self.__qc.measure(k+1, next(iteration))
                        elif sigma_ij[k]=='Z':
                            self.__qc.measure(k+1, next(iteration))
                        else: raise SyntaxError("unexpected measurement basis")
                    experiment = self.exp__qc(shots=shots)
                    exp_ij += experiment
                    # ここで 、いくつかあるsigma に対して和をとっている。つまり、このexp__qcを、statevectorに変えたら良いと思う。

            exp_ij_list.append(exp_ij)
        exp_i, exp_j = exp_ij_list
        return -exp_i*exp_j/4

    def W_i(self, params, del_theta, i, shots, statevector):
        '''
        a function to calculate W_i(correction term)
        input:
            params: list of parameters
            del_theta: difference of theta
            i: index
        output: value of W_i
        '''
        paulis, coeffs = self.hamiltonian(self.theta+ del_theta)
        temp = 0
        for j in range(len(paulis)):
            temp += -coeffs[j]/2*self.W_ij(params, i, paulis[j], shots, statevector)
        return temp

    def W_ij(self, params, i, sigma_j, shots, statevector):
        '''
        a function to calculate W_ij*(-2/h_j)
        #######CAUTION!#######
        This funtion is not calculating W_ij in the above equation!!
        #######CAUTION!#######
        input:
            params: list of parameters
            i: index i
            pauli_j: pauli_j of the hamiltonian
            shots: shots
        output:
            W_ij*(-2/h_j) = <sigma_i>0-->i-1<sigma_j>0-->N
        '''
        _, __, ___, sigma_i_list = self.ansatz_list[i]
        exp_list = []


        for j, sigma_list in [(i, sigma_i_list), (len(params), [sigma_j])]:
            exp_ij = 0

            for sigma_ij in sigma_list:
                num_cbit = self.num_qubits - sigma_ij.count('I')
                iteration = iter(range(num_cbit))
                ## ancilla bit is in __qc just for convention... only 1 --> self.num_qubits sites are used 
                self.__qc = QuantumCircuit(self.num_qubits+1, num_cbit)
                self.__qc.x(k for k in range(self.num_qubits+1) if k%2==1) # type: ignore
                for k in range(j):
                    name, place, fun, sigma = self.ansatz_list[k]
                    fun(place, params[k], name, sigma)
                if statevector:
                    est = Estimator()
                    sigma_est_ij = SparsePauliOp([sigma_ij[::-1]+"I"], coeffs=[1])
                    job = est.run(self.__qc, sigma_est_ij)
                    result = job.result()
                    exp_ij += result.values[0]

                else:
                    for k in range(self.num_qubits):
                        if sigma_ij[k] == "I": pass
                        elif sigma_ij[k] == "X":
                            self.__qc.h(k+1)
                            self.__qc.measure(k+1, next(iteration))
                        elif sigma_ij[k] == "Y":
                            self.__qc.sdg(k+1)
                            self.__qc.h(k+1)
                            self.__qc.measure(k+1, next(iteration))
                        elif sigma_ij[k]=='Z':
                            self.__qc.measure(k+1, next(iteration))
                        else: raise SyntaxError("unexpected measurement basis")
                    exp_ij += self.exp__qc(shots=shots)
            exp_list.append(exp_ij)
        return exp_list[0]*exp_list[1]

    def exp__qc(self, shots):
        '''
        get expectation value of self.__qc. all the qubits are measured in pauli basis
        積の形ならOK. suppose that circuit is already composed
        '''
        job = execute(self.__qc, self.backend, shots=shots)
        counts = job.result().get_counts()
        keys = list(counts.keys())
        prob = np.array(list(counts.values()))/shots
        for k in range(len(keys)):
            keys[k] = [*(keys[k][::-1])]
            for l in range(len(keys[k])):
                keys[k][l] = int(keys[k][l])
        exp = []
        for k in range(len(keys)):
            key = keys[k]
            ######## actual calculation #######
            temp = 1
            for l in range(len(key)):
                temp *= 1-2*key[l]
            exp.append(temp)
            ######## actual calculation #######
        exp = np.array(exp)
        return np.sum(exp*prob)
    
    def conditional_Z(self, place, param, _, __, put_XCX = False, put_C = False):
        '''
        与えられたパラメータをもとにself.__qcに対してRZ gateを入れる関数
        place: int 
            just index for quantum register
            starting from 0 to self.num_qubits - 1 (inclusive)
        param: float
            parameter! not list of parameter
        _: string
            always "Z", just to differentiate between odd and even type, so not used in this function
        __: string
            "ZIIII...", "IZIII...", "IIZII...",...
            not used in this function
        '''
        if not put_XCX and not put_C:
            self.__qc.rz(param, place+1)
        elif put_XCX:
            self.__qc.x(0)
            self.__qc.cz(0, place+1)
            self.__qc.x(0)
            self.__qc.rz(param, place+1)
        elif put_C:
            self.__qc.cz(0, place+1)

    def conditional_ZZ(self, place, param, name, _, put_XCX = False, put_C = False):
        '''
        与えられたパラメータをもとにself.__qcに対してRZZ gateを入れる関数。場合によって、conditional zz gate入れたりもする。
        place: int 
            just index for quantum register
            starting from 0 (depending on even or odd)
        param: float
            parameter! not list of parameter
        name: string
            either one of "even_ZZ" or "odd_ZZ"
        _: string
            "ZZII...", "IIZZI...",..., "IZZII...", ...
            not used in this function
        '''
        place_ind = place*2+1 if name=="even_ZZ" else place*2+2 if name=="odd_ZZ" else  exec("raise SyntaxError('unexpected name, ' + str(name))")

        if not put_XCX and not put_C: pass
        elif put_XCX:
            self.__qc.x(0)
            self.__qc.cz(0, place_ind)
            self.__qc.cz(0, place_ind+1)
            self.__qc.x(0)
        else:
            self.__qc.cz(0, place_ind)
            self.__qc.cz(0, place_ind+1)
            return
        self.__qc.cx(place_ind, place_ind+1)
        self.__qc.rz(param, place_ind+1)
        self.__qc.cx(place_ind, place_ind+1)

    def conditional_XXYY(self, place, param, name, sigma, put_XCX = False, put_C = False):
        '''
        与えられたパラメータをもとにself.__qcに対してexp(XX+YY) gateを入れる関数。場合によって、conditional XX or YY gate入れたりもする。
        place: int 
            just index for quantum register
            starting from 0 to self.num_qubits - 2 (inclusive)
        param: float
            parameter! not list of parameter
        name: string
            either one of "even_XXYY" or "odd_XXYY"
        sigma: string
            either "XX" or "YY" is in this string
        '''
        place_ind = place*2+1 if name=="even_XXYY" else place*2+2 if name=="odd_XXYY" else  exec("raise SyntaxError('unexpected name, ' + str(name))")

        if not put_XCX and not put_C: pass
        elif put_XCX:
            self.__qc.x(0)
            if "XX" in sigma:
                self.__qc.cx(0, place_ind)
                self.__qc.cx(0, place_ind+1)
            elif "YY" in sigma:
                self.__qc.cy(0, place_ind) # type: ignore
                self.__qc.cy(0, place_ind+1)
            else: raise SyntaxError("sigma cannot take this name,", sigma)
            self.__qc.x(0)
        else:
            if "XX" in sigma:
                self.__qc.cx(0, place_ind) # type: ignore
                self.__qc.cx(0, place_ind+1)
            elif "YY" in sigma:
                self.__qc.cy(0, place_ind)
                self.__qc.cy(0, place_ind+1)
            else: raise SyntaxError("sigma cannot take this name,", sigma)
            return
        self.__qc.cx(place_ind, place_ind+1)
        self.__qc.h(place_ind)
        self.__qc.cx(place_ind, place_ind+1)
        self.__qc.rz(param, place_ind)
        self.__qc.rz(-param, place_ind+1)
        self.__qc.cx(place_ind, place_ind+1)
        self.__qc.h(place_ind)
        self.__qc.cx(place_ind, place_ind+1)
    
    def save_params(self, path):

        np.save(path, self.params_list)

    def load_params(self, path):
        self.params_list = np.load(path)
        self.init_parameters = self.params_list[0]