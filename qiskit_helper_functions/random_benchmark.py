import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import CUGate, HGate, TGate, XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
import random

class RandomCircuit(object):
    def __init__(self, width, depth, connection_degree, seed) -> None:
        '''
        Generate a random benchmark circuit
        width: number of qubits
        depth: depth of the random circuit
        connection_degree: decay rate of number of direct contacts
        num_hadamards: number of H gates in the encoding layer. Overall number of solutions = 2^num_H
        '''
        super().__init__()
        random.seed(seed)
        self.width = width
        self.depth = depth
        # Decay rate of #targets
        self.num_targets_ubs = [self.width-1]
        for qubit in range(1,width):
            max_num_targets = connection_degree * self.num_targets_ubs[-1]
            max_num_targets = min(max_num_targets,self.width-1-qubit)
            max_num_targets = int(max_num_targets)
            self.num_targets_ubs.append(max_num_targets)
        # print('num_targets_ubs = {}'.format(self.num_targets_ubs))
    
    def generate(self):
        entangled_circuit, num_targets = self.generate_entangled()
        # print('%d 2q gates. %d tensor factors. %d depth.'%(
        #     entangled_circuit.num_nonlocal_gates(),
        #     entangled_circuit.num_unitary_factors(),
        #     entangled_circuit.depth()
        #     ))
        # print('num_targets = {}'.format(num_targets))
        return entangled_circuit
    
    def generate_entangled(self):
        entangled_circuit = QuantumCircuit(self.width,name='q')
        entangled_dag = circuit_to_dag(entangled_circuit)
        # for qubit in entangled_dag.qubits:
        #     entangled_dag.apply_operation_back(op=HGate(),qargs=[qubit],cargs=[])
    
        qubit_targets = {qubit:set() for qubit in range(self.width)}
        while True:
            '''
            Apply a random two-qubit gate to either left_dag or right_dag
            '''
            random_control_qubit_idx = self.get_random_control(qubit_targets)
            random_target_qubit_idx = self.get_random_target(random_control_qubit_idx,qubit_targets)
            
            random_control_qubit = entangled_dag.qubits[random_control_qubit_idx]
            random_target_qubit = entangled_dag.qubits[random_target_qubit_idx]
            theta = random.uniform(-math.pi,math.pi)
            phi = random.uniform(-math.pi,math.pi)
            lam = random.uniform(-math.pi,math.pi)
            gamma = random.uniform(-math.pi,math.pi)
            entangled_dag.apply_operation_back(op=CUGate(theta,phi,lam,gamma),qargs=[random_control_qubit,random_target_qubit],cargs=[])
            qubit_targets[random_control_qubit_idx].add(random_target_qubit_idx)

            '''
            Apply two random 1-q gates
            '''
            for _ in range(2):
                single_qubit_gate = random.choice([HGate(), TGate(), XGate(), YGate(), ZGate()])
                random_qubit = entangled_dag.qubits[random.choice(range(self.width))]
                entangled_dag.apply_operation_back(op=single_qubit_gate,qargs=[random_qubit],cargs=[])
            
            ''' Terminate when there is enough depth '''
            if entangled_dag.depth()>=self.depth:
                break
        entangled_circuit = dag_to_circuit(entangled_dag)
        num_targets = [len(qubit_targets[qubit]) for qubit in range(self.width)]
        for qubit in range(self.width):
            assert num_targets[qubit]<=self.num_targets_ubs[qubit]
        return entangled_circuit, num_targets

    def get_random_control(self, qubit_targets):
        '''
        Get a random control qubit
        Prioritize the ones with spare targets
        Else choose from qubits with #targets>0
        '''
        candidates = []
        for qubit in qubit_targets:
            if len(qubit_targets[qubit])<self.num_targets_ubs[qubit]:
                candidates.append(qubit)
        if len(candidates)>0:
            return random.choice(candidates)
        else:
            candidates = []
            for qubit, num_targets in enumerate(self.num_targets_ubs):
                if num_targets>0:
                    candidates.append(qubit)
            return random.choice(candidates)

    def get_random_target(self, control_qubit, qubit_targets):
        '''
        Get a random target qubit
        If the control qubit has exhausted its #targets, choose from existing targets
        Else prioritize the ones that have not been used
        '''
        if len(qubit_targets[control_qubit])<self.num_targets_ubs[control_qubit]:
            candidates = []
            for qubit in range(control_qubit+1,self.width):
                if qubit not in qubit_targets[control_qubit]:
                    candidates.append(qubit)
            return random.choice(candidates)
        else:
            return random.choice(list(qubit_targets[control_qubit]))