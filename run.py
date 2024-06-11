import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definindo a rede neural
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Função para treinar a rede neural
def train_model(model, criterion, optimizer, x_train, y_train, epochs):
    model.train()
    loss_values = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
    return model, loss_values

# Função para salvar o modelo
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Função para carregar o modelo
def load_model(path):
    model = RegressionModel()
    model.load_state_dict(torch.load(path))
    return model

# Parâmetros do treinamento
st.sidebar.header("Parâmetros do Treinamento")
epochs = st.sidebar.number_input("Número de Épocas", min_value=1, max_value=5000, value=100)
learning_rate = st.sidebar.number_input("Taxa de Aprendizado", min_value=0.0001, max_value=1.0, value=0.01)
save_path = st.sidebar.text_input("Caminho para Salvar o Modelo", "regression_model.pth")
load_path = st.sidebar.text_input("Caminho para Carregar o Modelo", "regression_model.pth")

# Configurações da função e dos dados
st.sidebar.header("Configurações da Função e Dados")
function_str = st.sidebar.text_input("Função para Predizer (usar 'np' para numpy)", "np.cos(x)")
input_range = st.sidebar.slider("Intervalo de Entrada (x)", -10.0, 10.0, (-2 * np.pi, 2 * np.pi))
num_points = st.sidebar.number_input("Número de Pontos", min_value=10, max_value=1000, value=100)

# Gerar dados de treino
x = np.linspace(input_range[0], input_range[1], num_points).reshape(-1, 1)
y = eval(function_str)

x_train = torch.Tensor(x)
y_train = torch.Tensor(y)

# Instanciar o modelo, critério e otimizador
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Treinando ou carregando o modelo
training_done = False
loss_values = []

if st.sidebar.button("Treinar Modelo"):
    model, loss_values = train_model(model, criterion, optimizer, x_train, y_train, epochs)
    st.write("Modelo treinado.")
    save_model(model, save_path)
    st.write(f"Modelo salvo em {save_path}")
    training_done = True

if st.sidebar.button("Carregar Modelo"):
    model = load_model(load_path)
    st.write(f"Modelo carregado de {load_path}")

# Avaliação do modelo
model.eval()
with torch.no_grad():
    predictions = model(x_train).numpy()

# Plotar os resultados
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Dados Originais")
plt.plot(x, predictions, color='red', label="Predições")
plt.legend()
plt.title("Regressão da Função")
plt.xlabel("x")
plt.ylabel("y")
st.pyplot(plt)

# Plotar a função de custo se o treinamento foi feito
if training_done:
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), loss_values, label="Custo")
    plt.legend()
    plt.title("Função de Custo durante o Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Custo (Loss)")
    st.pyplot(plt)
