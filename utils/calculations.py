import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

def calculate_returns(prices):
    """Calcular retornos diarios"""
    return prices.pct_change().dropna()

def calculate_annualized_volatility(returns):
    """Calcular volatilidad anualizada"""
    return returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calcular ratio de Sharpe"""
    excess_returns = returns - risk_free_rate/252
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def calculate_var(returns, confidence_level=0.95):
    """Calcular Value at Risk (VaR) histórico"""
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cagr(prices):
    """Calcular tasa de crecimiento anual compuesta (CAGR)"""
    days = (prices.index[-1] - prices.index[0]).days
    return ((prices.iloc[-1] / prices.iloc[0]) ** (365/days) - 1) * 100

def calculate_beta(asset_returns, market_returns):
    """Calcular beta de un activo"""
    covariance = np.cov(asset_returns, market_returns)[0][1]
    variance = np.var(market_returns)
    return covariance / variance

def calculate_capm(asset_returns, market_returns, risk_free_rate=0.02):
    """Calcular retorno esperado usando CAPM"""
    beta = calculate_beta(asset_returns, market_returns)
    market_return = np.mean(market_returns) * 252
    return risk_free_rate + beta * (market_return - risk_free_rate)

def markowitz_optimization(returns_df, num_portfolios=10000, risk_free_rate=0.02):
    """Optimización de portafolio usando el modelo de Markowitz"""
    cov_matrix = returns_df.cov() * 252
    expected_returns = returns_df.mean() * 252
    
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(len(returns_df.columns))
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return results, expected_returns, cov_matrix