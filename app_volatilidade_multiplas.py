
import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm

# Black-Scholes para call europeia
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Método de bisseção para encontrar a volatilidade implícita (Europeia)
def calcular_vol_europeia(preco_mercado, S, K, T, r, tol=1e-5, max_iter=100):
    low, high = 0.0001, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes_call(S, K, T, r, mid)
        if abs(price - preco_mercado) < tol:
            return mid
        elif price > preco_mercado:
            high = mid
        else:
            low = mid
    return mid

# Binomial Tree (Call Americana) + bisseção
def preco_binomial_americana(S, K, T, r, sigma, steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    prices = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            prices[j, i] = S * (u ** (i - j)) * (d ** j)

    values = np.maximum(prices[:, steps] - K, 0)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            early_exercise = max(prices[j, i] - K, 0)
            hold = disc * (p * values[j, i + 1] + (1 - p) * values[j + 1, i + 1])
            values[j, i] = max(early_exercise, hold)

    return values[0, 0]

def calcular_vol_americana(preco_mercado, S, K, T, r, tol=1e-4, max_iter=100):
    low, high = 0.0001, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = preco_binomial_americana(S, K, T, r, mid)
        if abs(price - preco_mercado) < tol:
            return mid
        elif price > preco_mercado:
            high = mid
        else:
            low = mid
    return mid

# Asiática (Monte Carlo)
def preco_asiatica_mc(S, K, T, r, sigma, n_sim=10000, steps=252):
    dt = T / steps
    prices = np.zeros((n_sim, steps))
    prices[:, 0] = S
    for t in range(1, steps):
        z = np.random.standard_normal(n_sim)
        prices[:, t] = prices[:, t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    avg_price = np.mean(prices, axis=1)
    payoff = np.maximum(avg_price - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

def calcular_vol_asiatica(preco_mercado, S, K, T, r, tol=1e-3, max_iter=50):
    low, high = 0.0001, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = preco_asiatica_mc(S, K, T, r, mid)
        if abs(price - preco_mercado) < tol:
            return mid
        elif price > preco_mercado:
            high = mid
        else:
            low = mid
    return mid

# Streamlit App
st.title("Volatilidade Implícita para Opções")

tipo_opcao = st.selectbox("Tipo de opção", ["Europeia", "Americana", "Asiática"])
ticker = st.text_input("Ticker do ativo (ex: AAPL, VALE)", value="AAPL")
K = st.number_input("Preço de exercício (strike)", value=150.0)
T = st.number_input("Tempo até vencimento (anos)", value=1.0)
r = st.number_input("Taxa de juros livre de risco", value=0.05)
preco_opcao = st.number_input("Preço de mercado da opção", value=10.0)

if st.button("Calcular Volatilidade Implícita"):
    with st.spinner("Buscando dados..."):
        try:
            S = float(yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1])
        except:
            st.error("Erro ao obter o preço do ativo.")
            st.stop()

        with st.spinner("Calculando volatilidade..."):
            if tipo_opcao == "Europeia":
                vol = calcular_vol_europeia(preco_opcao, S, K, T, r)
            elif tipo_opcao == "Americana":
                vol = calcular_vol_americana(preco_opcao, S, K, T, r)
            else:
                vol = calcular_vol_asiatica(preco_opcao, S, K, T, r)

        st.success(f"Volatilidade implícita ({tipo_opcao}): {vol * 100:.2f}%")
