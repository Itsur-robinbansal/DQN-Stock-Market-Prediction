# Deep Q-Network (DQN) for Stock Market Trading

This project implements a Deep Q-Network (DQN) based reinforcement learning agent that learns to trade stocks using historical data and technical indicators. It supports recurrent architectures (RNN/LSTM/GRU) and includes features for saving model outputs, downloading processed data, and generating visualizations.

---

## 🌐 Overview

This repo trains a DQN agent to trade Apple (AAPL) stock based on technical indicators. The agent learns to make buy/sell/hold decisions by interacting with a simulated trading environment using deep reinforcement learning.

### Key Features
- **RL algorithm:** Deep Q-Network (DQN)
- **Architectures:** RNN / LSTM / GRU
- **Data source:** Yahoo Finance via `yfinance`
- **Technical Indicators:** SMA, EMA, MACD, RSI, Bollinger Bands, VWAP, ATR, etc.
- **Actions:** 11 discrete trading actions (Sell 100% to Buy 100%)
- **Reward:** Portfolio value changes with penalties for overtrading & drawdown
- **Logging:** All training logs stored in `output_log.txt`
- **Output:** Portfolio performance metrics, equity curves, trading signal plots, Q-value evolution

---

## 📊 Performance Summary

From `output_log.txt`:
- **Training Period:** Jan 2020 - Mar 2022
- **Testing Period:** Oct 2021 - Mar 2022
- **Final Return (Testing):** Up to **+60.89%** total return
- **Baseline Comparison:** Moving average crossover ~8.45%

---

## 📚 Project Structure

```
├── index.py         # Main RL agent and training script
├── output_log.txt           # Training and testing logs
├── Results/
│   ├── Figures/             # Generated charts (signals, returns, actions)
├── Models/                  # Saved model weights (.pth)
├── Data/                    # Downloaded and processed stock data
```

---

## 📚 How It Works

### 1. Data Collection & Preprocessing
- Downloads AAPL daily OHLCV data (Yahoo Finance)
- Computes over 20 technical indicators
- Normalizes all features

### 2. Environment
- Simulates trading with initial cash, shares, and fees
- Tracks cash, positions, portfolio value, actions
- Provides reward based on performance

### 3. Agent (DQN)
- Recurrent network (RNN/LSTM/GRU) with dense layers
- Experience replay
- Target network updated periodically
- Epsilon-greedy action selection

### 4. Training & Evaluation
- Trained on monthly chunks (month-by-month training)
- Evaluated using:
  - Total Return
  - Sharpe Ratio
  - Max Drawdown
  - Win/Loss Ratio
  - Annualized Volatility

---

## 📈 Visualizations (Saved in `Results/Figures/`)
- Equity curve (`*_equity_curve.png`)
- Trading signals overlay on price chart (`*_trading_signals.png`)
- Q-values over time (`*_q_values.png`)
- Reward distribution histogram
- Action frequency bar chart

---

## 🔧 Requirements
```bash
pip install yfinance pandas numpy matplotlib torch ta
```

---

## ▶️ Running the Code
```bash
python index.py
```
All outputs (logs, data files, model weights, and visualizations) will be saved automatically.

---

## 🔄 Future Improvements
- Support for Double & Dueling DQN
- Multi-stock trading
- Live trading API integration
- Continuous online learning

---

## 📄 Authors
- Robin Bansal 
- robinkb200204@gmail.com

---

## 📅 License
This project is for academic purposes. Contact the authors for use in production or commercial settings.

