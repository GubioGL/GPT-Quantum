import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
from qutip import basis, mesolve, sigmax, sigmaz,sigmay, qeye,Options

def data_qubit_one(delta=1.0,tfinal=2, N=100, device="cpu",condiction=0):
    options = Options( store_states = True)
    H       = delta* np.pi * sigmax() 
    c_ops   = [qeye(2)]

    # Estado inicial
    psi0 = basis(2, 0)

    # Lista de tempos
    x_train = np.linspace(0, tfinal, N)

    # Operadores para medição <O>
    if condiction == 0:
        O_op = [sigmax()]
    elif condiction==1:
        O_op = [sigmay()]
    elif condiction==2:
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


# Define the loss function
def mse_loss(y_pred, y_true):
    return tc.mean((y_pred - y_true)**2)

# Define the loss function
def msa_loss(y_pred, y_true):
    return tc.mean(abs(y_pred - y_true))

def diagonal(M):
    traco_ = M.diagonal(offset=0,dim1=1, dim2=2)
    return traco_

def expected(A,B):
    return diagonal(A@B)

def commutator(A, B):
    return tc.matmul(A,B) - tc.matmul(B,A)

def Loss_EDO(H_,rho_rvetor,rho_ivetor,tempo,base_rho):
    rho_  = rho_rvetor.reshape((rho_rvetor.shape[0],base_rho,base_rho)) + 1j*rho_ivetor.reshape((rho_rvetor.shape[0],base_rho,base_rho))
    #  Calculando o Comutador do Hamiltoniano com a matriz densidade. 
    #   O resultado deve ser um tensor de 100 linhas e (2,2).
    H_rho_R = (commutator(H_.real,rho_.imag)- commutator(H_.imag,rho_.real))
    H_rho_I = (commutator(H_.imag,rho_.imag)- commutator(H_.real,rho_.real))
    #converter para vetor
    H_rho_R = H_rho_R.reshape((len(tempo),base_rho**2))
    H_rho_I = H_rho_I.reshape((len(tempo),base_rho**2))
    # Calculando o gradiente de drho_dt separando a parte real e imagina
    # Em seguida, iremos  calcular o Erro quadrático médio da equaçao de Von Neumann
    loss_edo = 0
    for i in range(rho_rvetor.shape[1]):
        drho_dt_real = tc.autograd.grad(outputs = rho_rvetor[:,i], 
                            inputs = tempo,
                            grad_outputs = tc.ones_like(rho_rvetor[:,i]),
                            retain_graph = True,
                            create_graph = True
                            )[0].reshape(rho_rvetor.shape[0])

        drho_dt_imag = tc.autograd.grad(outputs = rho_ivetor[:,i], 
                            inputs = tempo,
                            grad_outputs = tc.ones_like(rho_ivetor[:,i]),
                            retain_graph = True,
                            create_graph = True
                            )[0].reshape(rho_ivetor.shape[0])
        
        # Von Neuman equation
        loss_edo += tc.mean( (drho_dt_real - H_rho_R[:,i])**2 + (drho_dt_imag - H_rho_I[:,i])**2 )
    return loss_edo

def Loss_EDO2(H_,rho_rvetor,rho_ivetor,tempo,base_rho):
    rho_  = rho_rvetor.reshape((rho_rvetor.shape[0],base_rho,base_rho)) + 1j*rho_ivetor.reshape((rho_rvetor.shape[0],base_rho,base_rho))
    #  Calculando o Comutador do Hamiltoniano com a matriz densidade. 
    #   O resultado deve ser um tensor de 100 linhas e (2,2).
    H_rho_R = (commutator(H_.real,rho_.imag)- commutator(H_.imag,rho_.real))
    H_rho_I = (commutator(H_.imag,rho_.imag)- commutator(H_.real,rho_.real))
    #converter para vetor
    H_rho_R = H_rho_R.reshape((len(tempo),base_rho**2))
    H_rho_I = H_rho_I.reshape((len(tempo),base_rho**2))
    # Calculando o gradiente de drho_dt separando a parte real e imagina
    # Em seguida, iremos  calcular o Erro quadrático médio da equaçao de Von Neumann
    loss_edo = 0
    for i in range(rho_rvetor.shape[1]):
        drho_dt_real = tc.autograd.grad(outputs = rho_rvetor[:,i], 
                            inputs = tempo,
                            grad_outputs = tc.ones_like(rho_rvetor[:,i]),
                            retain_graph = True,
                            create_graph = True
                            )[0][:,0]

        drho_dt_imag = tc.autograd.grad(outputs = rho_ivetor[:,i], 
                            inputs = tempo,
                            grad_outputs = tc.ones_like(rho_ivetor[:,i]),
                            retain_graph = True,
                            create_graph = True
                            )[0][:,0]
        
        # Von Neuman equation
        loss_edo += tc.mean( (drho_dt_real - H_rho_R[:,i])**2 + (drho_dt_imag - H_rho_I[:,i])**2 )
    return loss_edo

def expected_plot( rho_,O_,expected_data,time_):
    v_esperados = expected(rho_, O_).sum(dim=-1).real

    plt.scatter(time_.detach().numpy(), v_esperados.detach().numpy(), c="r", marker=".", label="Neural Network")
    plt.scatter(time_.detach().numpy(), expected_data.detach().numpy(), c="g", marker=".", label="Data")
    plt.xlabel("Time")
    #plt.ylabel("<sigma_z>")
    plt.legend()
    plt.show()

def plots_rho(rho_NNR=0,rho_NNI=0,rho_data=0 ):
    fig, axs = plt.subplots(nrows=3, ncols=2 , figsize=(12,4), sharex=True)

    im =axs[0,0].imshow(rho_NNR.detach().numpy().T,cmap="jet")
    axs[0,0].set_title(r"$\mathcal{R}(\rho_{NN})$")
    axs[0,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[1,0].imshow(rho_data.real.T.detach().numpy(),cmap="jet")
    axs[1,0].set_title(r"$\mathcal{R}(\hat{\rho}_t)$")
    axs[1,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[2,0].imshow(abs(rho_data.real-rho_NNR).T.detach().numpy(),cmap="jet")
    axs[2,0].set_title(r"$|\mathcal{R}(\hat{\rho}_t) - \mathcal{R}(\rho_t)|$")
    axs[2,0].set_xlabel(r"$t$")
    axs[2,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[0,1].imshow(rho_NNI.detach().numpy().T,cmap="jet")
    axs[0,1].set_title(r"$\mathcal{I}(\rho_{NN})$")
    axs[0,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[1,1].imshow(rho_data.imag.T.detach().numpy(),cmap="jet")
    axs[1,1].set_title(r"$\mathcal{I}(\hat{\rho}_t)$")
    axs[1,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im = axs[2,1].imshow(abs(rho_data.imag-rho_NNI).T.detach().numpy() ,cmap="jet")
    axs[2,1].set_title(r"$|\mathcal{I}(\hat{\rho}_{NN}) - \mathcal{I}(\rho_t)|$")
    axs[2,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    plt.tight_layout()
    plt.show()

class SIN(nn.Module):
    def __init__(self): 
        super(SIN, self).__init__() 
    def forward(self, x):
        return tc.sin(x)
    
class Rede(nn.Module):
    def __init__(self, neuronio, activation,input_=1, output_=1, creat_p=False, N_of_paramater=1):
        super().__init__()
        self.neuronio   = neuronio
        self.output     = output_
        self.creat_p    = creat_p
        self.N_of_paramater = N_of_paramater
        
        # input camada linear
        self.hidden_layers = nn.ModuleList([nn.Linear(input_, neuronio[0])])
        # camadas do meio
        self.hidden_layers.extend([nn.Linear(neuronio[_], neuronio[_+1]) for _ in range(len(self.neuronio)-1)])
        # Última camada linear
        self.output_layer = nn.Linear(neuronio[-1], output_)

        # Função de ativação
        self.activation_ = activation
        # criar o parametro
        if creat_p:
            self.parametro = nn.Parameter(tc.rand(N_of_paramater)*2*np.pi)
            
    def forward(self, x):
        for layer,Activation in zip(self.hidden_layers,self.activation_) :
            x = Activation(layer(x))
        x = self.output_layer(x)
        return x

class Qubit:
    def __init__(self,base_rho_, device="cpu", neuro_=[10]):
        
        self.device = device
        self.N_neuronio = neuro_
        self.base_rho   = base_rho_

    def _initialize_networks(self,input_size=1,path=".../", load_net=False):
        self.real_net = Rede(neuronio   = self.N_neuronio,
                             input_     = input_size,
                             output_    = self.base_rho**2,
                             activation = [SIN()]*len(self.N_neuronio)
                             ).to(self.device)

        self.imag_net = Rede(neuronio   = self.N_neuronio,
                            input_      = input_size,
                            output_     = self.base_rho**2,
                            activation  = [SIN()] * len(self.N_neuronio)
                            ).to(self.device)
        
        if load_net == True:
            #self.real_net = tc.load(path)['real_net']
            #self.imag_net = tc.load(path)['imag_net']
            self.real_net.load_state_dict(tc.load(path)['real_net'])
            self.imag_net.load_state_dict(tc.load(path)['imag_net'])
            
            self.real_net.to(device=self.device)
            self.imag_net.to(device=self.device)

    def plot_evaluate(self,delta,N_,condiction_, time0=0,time1=1): 
        rho_train, expect,Hamilt,Observavel  = data_qubit_one(
                                                        delta   = delta[0].detach().numpy(),
                                                        tfinal  = time1, 
                                                        N       = N_,
                                                        device  = self.device,
                                                        condiction=condiction_)

        time    = tc.linspace(time0,time1,N_,dtype=tc.float32).reshape((-1, 1))
        deltalista_ = delta.repeat_interleave(N_).view(-1,1)
        inputs      = tc.cat((time, deltalista_),axis=1)
        
        self.real_net.eval()
        self.imag_net.eval()

        y_pred_real = self.real_net(inputs)
        y_pred_imag = self.imag_net(inputs)
        
        #plots_rho(y_pred_real, y_pred_imag, rho_train)
        self.rho = (y_pred_real + 1j * y_pred_imag).reshape((N_, self.base_rho, self.base_rho))
        self.Observavel = Observavel
        self.expect = expect
        self.time = time
        #expected_plot(rho_=rho,O_=Observavel,expected_data=expect,time_=time)

