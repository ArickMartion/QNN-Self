import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BarrierGate,Z,X,gene_univ_parameterized_gate,gene_univ_two_params_gate,RY,RZ
from mindquantum.simulator import Simulator
from mindquantum.core.parameterresolver import PRGenerator


def U_univ_parameterized_gate(name = "P1"):
    """Function:
    - Define a quantum photonic gate with a single parameter, i.e., a structure with one heater.
    - name ∈ ["P1", "P2", "M1", "M2"]: 1 (2) indicates the heater is on the top (bottom) of the waveguide; 
      "P" or "M" represents either two straight waveguides or one MZI.
    """
    
    if name not in ["P1","P2","M1","M2","iswap"]:
        return None
    
    U_mmi = np.array([[1,1j],
                  [1j,1]])/np.sqrt(2)
    
    def matrix(x):
        # Define the phase value
        p = np.exp(1j * x)
        
        if name[1] == "1":
            U = np.array([[p,0.0j],
                         [0.0j,1.0]])
        elif name[1] == "2":
            U = np.array([[1.0,0.0j],
                         [0.0j,p]])
        if name[0] == "M":
            U = U_mmi @ U @ U_mmi
        return U
    
    def diff_matrix(x):
        # Define the phase value
        dp = 1j*np.exp(1j * x)
        if name[1] == "1":
            dU = np.array([[dp,0.0j],
                         [0.0j,0.0]])
        elif name[1] == "2":
            dU = np.array([[0.0,0.0j],
                         [0.0j,dp]])
        if name[0] == "M":
            dU = U_mmi @ dU @ U_mmi
        return dU
    
    return gene_univ_parameterized_gate(name, matrix, diff_matrix)


def U_iswap():
    """
Function:
    - Define an parameterised iSWAP-like gate, which acts on the |01> and |10> states; it is somewhat special.
"""
    U_mmi = np.array([[1.0,0.0j,0.0j,0.0j],
                     [0.0j,1.0/np.sqrt(2),1j/np.sqrt(2),0.0j],
                     [0.0j,1.0j/np.sqrt(2),1.0/np.sqrt(2),0.0j],
                     [0.0j,0.0j,0.0j,1.0]])
    
    def matrix(x):
        # Define the phase value
        p = np.exp(1j * x)
        U = np.array([[1.0,0.0j,0.0j,0.0j],
                     [0.0j,p,0.0j,0.0j],
                     [0.0j,0.0j,1.0,0.0j],
                     [0.0j,0.0j,0.0j,1.0]])
        U = U_mmi @ U @ U_mmi
        return U
    
    def diff_matrix(x):
        # Define the phase value
        dp = 1j * np.exp(1j * x)
        dU = np.array([[0.0j,0.0j,0.0j,0.0j],
                     [0.0j,dp,0.0j,0.0j],
                     [0.0j,0.0j,0.0j,0.0j],
                     [0.0j,0.0j,0.0j,0.0j]])
        dU = U_mmi @ dU @ U_mmi
        return dU
    
    return gene_univ_parameterized_gate("iswap", matrix, diff_matrix)
    


def U_two_params_gate(name = "U_pre1"):
    """
    Function:
        - Define a quantum photonic gate with two parameters, i.e., a structure with two heaters.
        - name ∈ ["U_pre1", "U_pre2", "U_post1", "U_post2", "iswap"]: 1 (2) indicates the heaters are on the top (bottim) of the waveguide.
        - "pre" ("post") indicates one heater is before (after) the MZI.
    """
    
    if name not in ["U_pre1","U_pre2","U_post1","U_post2"]:
        return None
    
    U_mmi = np.array([[1,1j],
                  [1j,1]])/np.sqrt(2)
    
    def matrix(a,b):
        # Define the phase value
        pa = np.exp(1j * a)
        pb = np.exp(1j * b)
        
        if name[-1] == "1":
            Ua = np.array([[pa,0.0j],
                         [0.0j,1.0]])
            Ub = np.array([[pb,0.0j],
                         [0.0j,1.0]])
            
        elif name[-1] == "2":
            Ua = np.array([[1.0,0.0j],
                         [0.0j,pa]])
            Ub = np.array([[1.0,0.0j],
                         [0.0j,pb]])
            
        if name[2:5] == "pre":
            U = U_mmi @ Ub @ U_mmi @ Ua
        elif name[2:5] == "pos":
            U = Ub @ U_mmi @ Ua @ U_mmi
        return U
    
    def diff_matrix1(a,b):
        # Define the phase value
        dpa = 1j*np.exp(1j * a)
        pb = np.exp(1j * b)
        
        if name[-1] == "1":
            dUa = np.array([[dpa,0.0j],
                         [0.0j,0.0]])
            Ub = np.array([[pb,0.0j],
                         [0.0j,1.0]])
            
        elif name[-1] == "2":
            dUa = np.array([[0.0,0.0j],
                         [0.0j,dpa]])
            Ub = np.array([[1.0,0.0j],
                         [0.0j,pb]])
            
        if name[2:5] == "pre":
            dU = U_mmi @ Ub @ U_mmi @ dUa
        elif name[2:5] == "pos":
            dU = Ub @ U_mmi @ dUa @ U_mmi
        return dU
    
    def diff_matrix2(a,b):
        # Define the phase value
        pa = np.exp(1j * a)
        dpb = 1j*np.exp(1j * b)
        
        if name[-1] == "1":
            Ua = np.array([[pa,0.0j],
                         [0.0j,1.0]])
            dUb = np.array([[dpb,0.0j],
                         [0.0j,0.0]])
            
        elif name[-1] == "2":
            Ua = np.array([[1.0,0.0j],
                         [0.0j,pa]])
            dUb = np.array([[0.0,0.0j],
                         [0.0j,dpb]])
            
        if name[2:5] == "pre":
            dU = U_mmi @ dUb @ U_mmi @ Ua
        elif name[2:5] == "pos":
            dU = dUb @ U_mmi @ Ua @ U_mmi
        return dU
    
    return gene_univ_two_params_gate(name, matrix, diff_matrix1, diff_matrix2)


def U4(prg, qr = [0,1]):
    """
    Function: Define the quantum gate circuit that implements U4
    """
    
    qc = Circuit()
    
    # 1.3 phases
    qc += X.on(qr[1])
    qc += G_P1(prg.new()).on(qr[0],qr[1])
    qc += G_P2(prg.new()).on(qr[0],qr[1])
    qc += X.on(qr[1])
    qc += G_P1(prg.new()).on(qr[0],qr[1])
    
    # 2.2 MZIs
    qc += X.on(qr[1])
    qc += G_U_post1( prg.new(),prg.new() ).on(qr[0],qr[1])
    qc += X.on(qr[1])
    qc += G_U_post1( prg.new(),prg.new() ).on(qr[0],qr[1])
    
    # 3.iswap
    qc += G_iswap(prg.new()).on([qr[0],qr[1]])
    qc += X.on(qr[1])
    qc += G_P2(prg.new()).on(qr[0],qr[1])
    qc += X.on(qr[1])
    qc += G_P2(prg.new()).on(qr[0],qr[1])
    
    # 4.2 MZIs
    qc += X.on(qr[1])
    qc += G_U_post1( prg.new(),prg.new() ).on(qr[0],qr[1])
    qc += X.on(qr[1])
    qc += G_U_post1( prg.new(),prg.new() ).on(qr[0],qr[1])
    
    # 5.2 iswap
    qc += G_iswap(prg.new()).on([qr[0],qr[1]])
    qc += X.on(qr[1])
    qc += G_P2(prg.new()).on(qr[0],qr[1])
    qc += X.on(qr[1])
    qc += G_P2(prg.new()).on(qr[0],qr[1])
    
    return qc
    
    
def qc_Hidden_Layer1(prg):
    qc = Circuit()

    # 1.pump_distribution
    qc += G_U_post1( prg.new(),prg.new() ).on(3)
    qc += X.on(3)
    qc += G_U_post1( prg.new(),prg.new() ).on(1,3)
    qc += X.on(3)
    qc += G_U_post1( prg.new(),prg.new() ).on(1,3)
    qc += BarrierGate()

    # 2.Controlled Unitary

    # 00 control-qubits
    qc += X.on(1)
    qc += X.on(3)
    qc += G_U_post1( prg.new(),prg.new() ).on(0,[1,3])
    qc += G_U_post1( prg.new(),prg.new() ).on(2,[1,3])
    qc += X.on(1)
    qc += X.on(3)

    # 01 control-qubits
    qc += X.on(3)
    qc += G_U_post1( prg.new(),prg.new() ).on(0,[1,3])
    qc += G_U_post2( prg.new(),prg.new() ).on(2,[1,3])
    qc += X.on(3)

    # 10 control-qubits
    qc += X.on(1)
    qc += G_U_post2( prg.new(),prg.new() ).on(0,[1,3])
    qc += G_U_post1( prg.new(),prg.new() ).on(2,[1,3])
    qc += X.on(1)

    # 11 control-qubits
    qc += G_U_post2( prg.new(),prg.new() ).on(0,[1,3])
    qc += G_U_post2( prg.new(),prg.new() ).on(2,[1,3])
    qc += BarrierGate()
    
    return qc


def qc_Hidden_Layer2(prg, CCCZ = True):
    
    qc = Circuit()
    
    # 3.Two U4
    qc += U4(prg, qr = [0,1])
    qc += U4(prg, qr = [2,3])
    qc += BarrierGate()

    # 4.CCCZ gate
    if CCCZ == True:
        qc += X.on(2)
        qc += X.on(3)
        qc += Z.on(0,[1,2,3])
        qc += X.on(2)
        qc += X.on(3)
        qc += BarrierGate()

    # 4.CRY moule
    qc += G_M1(prg.new()).on(0,1)
    qc += X.on(1)
    qc += G_M1(prg.new()).on(0,1)
    qc += X.on(1)
    qc += G_M1(prg.new()).on(1,0)
    qc += X.on(0)
    qc += G_M1(prg.new()).on(1,0)
    qc += X.on(0)

    qc += G_M1(prg.new()).on(2,3)
    qc += X.on(3)
    qc += G_M1(prg.new()).on(2,3)
    qc += X.on(3)
    qc += G_M1(prg.new()).on(3,2)
    qc += X.on(2)
    qc += G_M1(prg.new()).on(3,2)
    qc += X.on(2)
    
    qc += BarrierGate()
    
    return qc


def state_tomography(prg):
    
    qc = Circuit()
    
    for qr in [[0,1],[2,3]]:
        qc += X.on(qr[1])
        qc += G_P1(prg.new()).on(qr[0],qr[1])
        qc += G_P2(prg.new()).on(qr[0],qr[1])
        qc += X.on(qr[1])
        qc += G_P1(prg.new()).on(qr[0],qr[1])
        
        qc += X.on(qr[1])
        qc += G_M1(prg.new()).on(qr[0],qr[1])
        qc += X.on(qr[1])
        qc += G_M1(prg.new()).on(qr[0],qr[1])
        
        qc += X.on(qr[1])
        qc += G_P2(prg.new()).on(qr[0],qr[1])
        qc += X.on(qr[1])
        qc += G_iswap(prg.new()).on([qr[0],qr[1]])
        
    return qc



def qc_chip(Enc1 = True, Enc2 = True, Hidden_1 = True, Hidden_2 = True, CCCZ = True, tomography = True):
    """
    Function:
        - Define the complete chip circuit
    """
    
    qc = Circuit()
    
    prg_enc = PRGenerator("a")
    prg = PRGenerator("x")
    
    # 1. Add input layer
    count_e = 0
    if Enc1 == True:
        for q in range(4):
            qc += RY(prg_enc.new()).on(q)
            count_e += 1
        qc += BarrierGate()
        
    
    # 2. Add hidden layer 1
    if Hidden_1 == True:
        qc += qc_Hidden_Layer1(prg)
    
    # 3. Add re-uploading layer
    if Enc1 == True:
        for q in range(4):
            qc += RZ(prg_enc.new()).on(q)
            count_e += 1
        qc += BarrierGate()
        
    # 4. Add hidden layer 1
    if Hidden_2 == True:
        qc += qc_Hidden_Layer2(prg, CCCZ)
    
    # 5. Add state tomography layer
    if tomography == True:
        qc += state_tomography(prg)
    
    return qc


#"""""""""""""""# Define the gates used""""""""""""""""""""""
G_P1 = U_univ_parameterized_gate("P1")
G_P2 = U_univ_parameterized_gate("P2")
G_M1 = U_univ_parameterized_gate("M1")
G_M2 = U_univ_parameterized_gate("M2")

G_U_pre1 = U_two_params_gate(name = "U_pre1")
G_U_pre2 = U_two_params_gate(name = "U_pre2")
G_U_post1 = U_two_params_gate(name = "U_post1")
G_U_post2 = U_two_params_gate(name = "U_post2")

G_iswap = U_iswap()