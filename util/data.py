import matplotlib.pyplot as plt
import numpy as np
import torch as tc
from qutip import basis, mesolve, sigmax, sigmaz, qeye,Options

def data_qubit_one(delta=1.0,tfinal=2, N=100, device="cpu"):
    options = Options( store_states = True)
    H       = delta* np.pi * sigmax() 
    c_ops   = [qeye(2)]

    # Estado inicial
    psi0 = basis(2, 0)

    # Lista de tempos
    x_train = np.linspace(0, tfinal, N)

    # Operadores para medição <O>
    O_op = [sigmaz()]

    # Executar simulação
    result = mesolve(H, psi0, x_train, c_ops, O_op,options=options)

    # Convertendo para numpy
    H       = np.array( H.full(),
                        dtype=np.complex64)
    O_op    = np.array( [op.full() for op in O_op],
                        dtype=np.complex64 )
    
    y_train = np.array( [ result.states[i].full().flatten() for i in range(N)],
                        dtype=np.complex64 )
    expect  = np.array( result.expect[0].real).reshape((-1, 1))

    H       = tc.tensor(H,device=device)
    O_op    = tc.tensor(O_op, device=device)
    y_train = tc.tensor(y_train, requires_grad=True, device=device)
    expect  = tc.tensor(expect, device=device)

    return  y_train, expect, H, O_op


# # Executar simulação
# tlist,state,res,_,_ = data_qubit_one(tfinal=2, N=100, device="cpu")

# # Plotar resultados
# plt.scatter(tlist.detach().numpy(), res.detach().numpy(), c="b", marker=".", label=r"$\langle \sigma_z \rangle$")
# plt.xlabel("Time")
# plt.ylabel(r"$\langle O \rangle$")
# plt.legend()
# plt.show()