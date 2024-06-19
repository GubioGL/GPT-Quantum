import streamlit as st
import torch as tc
import numpy as np
import matplotlib.pyplot as plt
from util.function import Qubit
from util.function import expected
st.sidebar.markdown("<h1 style='text-align: center; font-size: 2.5em;'>Hamiltoniana de um qubit</h1>", unsafe_allow_html=True)

#MARKDOWN()
st.sidebar.markdown("""
<div font-size: 1.2em;'>
    <p>System setup: </p>
    <p>We will start with a basic Hamiltonian for the qubit, which flips the state of the qubit.</p>
</div> """, unsafe_allow_html=True)

st.sidebar.latex(r"H = \frac{\omega}{2} \sigma_x")

st.sidebar.markdown("""
<div font-size: 1.2em;'>
    <p>onde</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.latex(r"""
\sigma_x = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
""")

st.sidebar.markdown("""
<div  font-size: 1.2em;'>
    <p>Master equation</p>
</div>""", unsafe_allow_html=True)

st.sidebar.latex(r"\frac{d}{dt}\rho= -i[H,\rho]")
st.sidebar.markdown("""  <div  font-size: 1.2em;'> where  rho is the output of Neural Network.  </div>""", unsafe_allow_html=True)

st.header("Parâmetros do qubits")


freq = st.number_input("Frequência (Versão beta escolha entre 0.5,0.75,1.0)",
                               min_value=0.5, max_value=1.3, value=0.5)
#dissipation = st.number_input("Dissipação (Versão beta dissipação = 0)", 
#                              min_value=0.0, max_value=1.0, value=0.0)
operator = st.radio("Operador para o valor esperado: ", ["Pauli-x", "Pauli-y", "Pauli-z"])
time = st.slider("Intervalo de tempo ", 0.0, 1.0, (0.0, 1.0))

if operator == "Pauli-x":
    operador = 0
elif operator == "Pauli-y":
    operador = 1
elif operator == "Pauli-z":
    operador = 2

# Gerando dados de teste

model = Qubit(base_rho_=2,neuro_=[50,50],device="cpu")
model._initialize_networks(input_size=2,load_net=True,path="util/parametro.pt")
model.plot_evaluate(condiction_=operador,delta=tc.tensor([freq]),N_=50,time0=time[0],time1=time[1])

st.markdown("## Resultados")
plt.figure(figsize=(8, 5))
v_esperados = expected(model.rho, model.Observavel).sum(dim=-1).real

plt.scatter(model.time.detach().numpy(), v_esperados.detach().numpy(), c="r", marker=".", label="Neural Network")
plt.scatter(model.time.detach().numpy(), model.expect.detach().numpy(), c="g", marker=".", label="qutip")
plt.xlabel("Time")
plt.ylabel("<sigma_z>")
plt.legend()
st.pyplot(plt)