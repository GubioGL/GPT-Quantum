import streamlit as st
import torch as tc
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Definição da rede neural
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)
    
    def forward(self, x):
        x = tc.tanh(self.hidden(x))
        x = self.output(x)
        return x

# Função para treinar a rede neural
def train_model2():
    st.title("Treinamento da Rede Neural")

    st.markdown("## Parâmetros de Treinamento")
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider('Taxa de Aprendizado', 0.001, 0.1, 0.01)
    with col2:
        epochs = st.slider('Épocas', 100, 1000, 500)

    input_range = st.slider('Range de Valores de Entrada', -10.0, 10.0, (-2.0 * np.pi, 2.0 * np.pi))

    # Gerando dados de treino
    x_train = np.linspace(input_range[0], input_range[1], 100).reshape(-1, 1).astype(np.float32)
    y_train = np.sin(x_train)

    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treinamento
    losses = []
    for epoch in range(epochs):
        model.train()
        inputs = tc.tensor(x_train)
        targets = tc.tensor(y_train)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Salvando o modelo
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    st.markdown("## Resultados do Treinamento")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(losses)
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Perda")
    ax1.set_title("Perda durante o treinamento")

    st.success("Modelo treinado e salvo com sucesso!")

    # Visualizando resultados do treinamento
    model.eval()
    with tc.no_grad():
        predictions = model(tc.tensor(x_train)).numpy()

    ax2.plot(x_train, y_train, label='Seno Real')
    ax2.plot(x_train, predictions, label='Predição', linestyle='dashed')
    ax2.legend()
    ax2.set_title("Valores reais vs Preditos")

    st.pyplot(fig)

# Função para carregar e usar um modelo treinado
def use_model2():
    st.title("Usar Modelo Treinado")

    # Carregando o modelo
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    input_range = st.slider('Range de Valores de Entrada', -10.0, 10.0, (-2.0 * np.pi, 2.0 * np.pi))

    # Gerando dados de teste
    x_test = np.linspace(input_range[0], input_range[1], 100).reshape(-1, 1).astype(np.float32)
    y_test = np.sin(x_test)

    model.eval()
    with tc.no_grad():
        inputs = tc.tensor(x_test)
        predictions = model(inputs).numpy()

    st.markdown("## Resultados")
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, y_test, label='Seno Real')
    plt.plot(x_test, predictions, label='Predição', linestyle='dashed')
    plt.legend()
    st.pyplot(plt)

import streamlit as st

# Define a função para cada opção da Sidebar
def train_model():
    st.write("Treinar Modelo")

def use_model():
    st.write("Carregar Modelo")

# Sidebar com opções de seleção
st.sidebar.title("Menu")
page = st.sidebar.radio("Selecione uma opção:", ("Treinar Modelo", "Carregar Modelo"))

# Chamada da função correspondente à opção selecionada
if page == "Treinar Modelo":
    train_model()
    train_model2()
elif page == "Carregar Modelo":
    use_model()
    use_model2()
