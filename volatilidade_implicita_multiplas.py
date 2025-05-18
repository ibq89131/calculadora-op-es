
import numpy as np
from scipy.stats import norm
import yfinance as yf

# Função de Black-Scholes para opção call europeia
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# Cálculo da volatilidade implícita via método da bisseção
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

# Parâmetros de exemplo para AAPL
ticker = "AAPL"
K = 190  # Strike
T = 0.5  # Tempo até o vencimento (em anos)
r = 0.05  # Taxa de juros anual
preco_mercado = 14.5  # Preço da call europeia no mercado

# Pegar último preço do ativo
dados = yf.download(ticker, period='1y')
S = float(dados['Close'].iloc[-1])

# Calcular volatilidade implícita para opção Europeia
vol_europeia = calcular_volatilidade_implicita(preco_mercado, S, K, T, r)
print(f"Volatilidade implícita (Europeia): {vol_europeia*100:.2f}%")

# Estimativa de volatilidade implícita para Americanas e Asiáticas (assumindo Black-Scholes como proxy)
print(f"Volatilidade implícita estimada (Americana): {vol_europeia*100:.2f}%")
print(f"Volatilidade implícita estimada (Asiática): {vol_europeia*100:.2f}%")
