import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# Technical Analysis - TA-Lib
from talib import RSI # https://github.com/ta-lib/ta-lib-python

# API da Yahoo Finance
import yfinance as yf

# machine learning
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler # para normalizar os dados
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

