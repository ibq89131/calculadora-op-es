
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

# Função para capturar dados de mercado
def capturar_parametros(ticker, periodo='1y'):
    dados = yf.download(ticker, period=periodo)
    dados['Retornos'] = np.log(dados['Close'] / dados['Close'].shift(1))
    dados = dados.dropna()
    S0 = float(dados['Close'].iloc[-1])
    mu = float(dados['Retornos'].mean() * 252)
    sigma = float(dados['Retornos'].std() * np.sqrt(252))
    return S0, mu, sigma, dados

# Modelos de opções
def calcular_opcao_europeia(S0, K, T, r, sigma, n_sim):
    Z = np.random.standard_normal(n_sim)
    ST = float(S0) * np.exp((float(r) - 0.5 * float(sigma) ** 2) * float(T) + float(sigma) * np.sqrt(float(T)) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-float(r) * float(T)) * np.mean(payoff)

def calcular_opcao_americana(S0, K, T, r, sigma, tipo='call', n=100):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    ST = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            ST[j, i] = S0 * (u ** (i - j)) * (d ** j)
    V = np.zeros_like(ST)
    if tipo == 'call':
        V[:, n] = np.maximum(0, ST[:, n] - K)
    else:
        V[:, n] = np.maximum(0, K - ST[:, n])
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            exercicio = 0
            if tipo == 'call':
                exercicio = max(0, ST[j, i] - K)
            else:
                exercicio = max(0, K - ST[j, i])
            V[j, i] = max(exercicio, np.exp(-r * dt) * (p * V[j, i+1] + (1 - p) * V[j+1, i+1]))
    return V[0, 0]

def calcular_opcao_asiatica(S0, K, T, r, sigma, n_sim=10000, steps=252):
    dt = T / steps
    S = np.zeros((n_sim, steps))
    S[:, 0] = S0
    for t in range(1, steps):
        Z = np.random.standard_normal(n_sim)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    media = np.mean(S, axis=1)
    payoff = np.maximum(media - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


# Interface Streamlit
st.title("Calculadora de Opções: Europeia, Americana e Asiática")
tipo_opcao = st.selectbox("Tipo de opção", ["Europeia", "Americana", "Asiática"])
ticker = st.text_input("Código do ativo (ex: AAPL)", "AAPL")
K = st.number_input("Preço de exercício (Strike)", value=150.0)
T = st.number_input("Tempo até o vencimento (em anos)", value=1.0)
r = st.number_input("Taxa de juros livre de risco (anual)", value=0.05)

if tipo_opcao == "Americana":
    tipo = st.selectbox("Tipo (call ou put)", ["call", "put"])
    passos = st.slider("Passos na árvore binomial", 10, 500, 100)
elif tipo_opcao == "Asiática":
    steps = st.slider("Passos simulados por ano", 50, 365, 252)
    n_sim = st.slider("Número de simulações", 1000, 50000, 10000)
else:
    n_sim = st.slider("Número de simulações", 1000, 50000, 10000)

if st.button("Calcular"):
    with st.spinner("Calculando..."):
        S0, mu, sigma, dados = capturar_parametros(ticker)
        if tipo_opcao == "Europeia":
            preco = calcular_opcao_europeia(S0, K, T, r, sigma, n_sim)
        elif tipo_opcao == "Americana":
            preco = calcular_opcao_americana(S0, K, T, r, sigma, tipo=tipo, n=passos)
        else:
            preco = calcular_opcao_asiatica(S0, K, T, r, sigma, n_sim=n_sim, steps=steps)
        st.success(f"Preço estimado da opção {tipo_opcao.lower()}: ${preco:.2f}")
