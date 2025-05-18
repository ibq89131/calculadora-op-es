
import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf

# Black-Scholes para call europeia
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# Volatilidade implícita via bisseção
def calcular_volatilidade_implicita(preco_mercado, S, K, T, r, tol=1e-5, max_iter=100):
    sigma_low = 0.0001
    sigma_high = 5.0
    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        preco_estimado = black_scholes_call(S, K, T, r, sigma_mid)
        if abs(preco_estimado - preco_mercado) < tol:
            return sigma_mid
        elif preco_estimado > preco_mercado:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return sigma_mid

# Streamlit app
st.title("Calculadora de Volatilidade Implícita (Opção Europeia Call)")

ticker = st.text_input("Ticker da ação (ex: AAPL)", value="AAPL")
K = st.number_input("Preço de exercício (Strike)", value=190.0)
T = st.number_input("Tempo até o vencimento (em anos)", value=0.5)
r = st.number_input("Taxa de juros livre de risco (anual)", value=0.05)
preco_mercado = st.number_input("Preço de mercado da opção (call)", value=14.5)

if st.button("Calcular volatilidade implícita"):
    with st.spinner("Buscando dados e calculando..."):
        dados = yf.download(ticker, period='1y')
        S = float(dados['Close'].iloc[-1])
        vol = calcular_volatilidade_implicita(preco_mercado, S, K, T, r)
        st.success(f"Volatilidade implícita estimada: {vol * 100:.2f}%")
