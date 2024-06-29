import streamlit as st
import matplotlib.pyplot as plt
def  twoqubit_open():
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 2.5em;'> Hamiltoniana do modelo Jaynes-Cummings</h1>", unsafe_allow_html=True)
    #MARKDOWN()
    st.sidebar.markdown("""
    <div font-size: 1.2em;'>
        <p>System setup: </p>
    </div> """, unsafe_allow_html=True)
    st.sidebar.latex(r"H_{JC} = \hbar \omega a^\dagger a + \frac{1}{2} \hbar \omega_0 \sigma_z + \hbar g (a \sigma_+ + a^\dagger \sigma_-)")

    st.sidebar.markdown("""
    <div font-size: 1.2em;'>
        <p>onde</p>  
    <ul>
        <li>\(\omega\) é a frequência do modo do campo.</li>
        <li>\(a\) e \(a^\dagger\) são os operadores de aniquilação e criação do fóton, respectivamente.</li>
        <li>\(\omega_0\) é a frequência de transição do átomo de dois níveis.</li>
        <li>\(\sigma_z\) é o operador de Pauli \(z\).</li>
        <li>\(\sigma_+\) e \(\sigma_-\) são os operadores de subida e descida do átomo, respectivamente.</li>
        <li>\(g\) é a força de acoplamento entre o átomo e o campo.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div  font-size: 1.2em;'>
        <p>Master equation</p>
    </div>""", unsafe_allow_html=True)
    st.sidebar.latex(r"\frac{d\rho}{dt} = -\frac{i}{\hbar} [H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \left\{ L_k^\dagger L_k, \rho \right\} \right)")
    st.sidebar.markdown("""  <div  font-size: 1.2em;'> where  rho is the output of Neural Network.  </div>""", unsafe_allow_html=True)

    st.pyplot(plt)