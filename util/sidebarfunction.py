import streamlit as st

def SIDEBAR():
    # Parâmetros do treinamento
    # Add a slider to the sidebar:
    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0))

    st.sidebar.header("Parâmetros do qubits")
    freq = st.sidebar.number_input("Frequência", min_value=-1.0, max_value=1.9, value=0.1)
    dissipation = st.sidebar.number_input("Dissipação", min_value=0.0, max_value=1.0, value=0.1)

    valorexperado = st.sidebar.number_input("Valor experado \n (1 - paulox , 2 - pauloy ,3- Pauliz )", min_value=1, max_value=3, value=1)
    time = st.sidebar.slider("Intervalo de tempo ", 0.0, 10.0, (0.0, 10.0))


