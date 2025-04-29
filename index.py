import collections
import warnings
import os
from collections import deque, namedtuple
from random import random, randrange

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
import matplotlib.pyplot as plt
import yfinance as yf
import ta  # Technical analysis library
import sys

# Redirect all prints to a log file
sys.stdout = open('output_log.txt', 'w')


device = device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('Models', exist_ok=True)
os.makedirs('Data', exist_ok=True)
os.makedirs('Results', exist_ok=True)

# Define namedtuple for transitions
Transition = namedtuple('Transition',
                        ('state', 'hidden_state1', 'hidden_state2', 'action', 'next_state', 'reward',
                         'next_hidden_state1', 'next_hidden_state2'))

# DQN Model Implementation
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, architecture='LSTM', 
                 dense_layers=2, dense_size=128, dropout_rate=0.2):
        super(DQN, self).__init__()
        self.architecture = architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.dense_layers_num = dense_layers
        self.dense_size = dense_size
        self.dropout_rate = dropout_rate
        
        # Network architecture based on specified type
        if architecture == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                               num_layers=1, batch_first=True)
        elif architecture == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                              num_layers=1, batch_first=True)
        else:  # Default to RNN
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, 
                              num_layers=1, batch_first=True)
        
        # Dense layers for processing RNN output
        self.dense_layers = nn.ModuleList()
        current_size = hidden_size
        
        for i in range(dense_layers):
            self.dense_layers.append(nn.Linear(current_size, dense_size))
            current_size = dense_size
        
        self.output_layer = nn.Linear(current_size, num_actions)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, hidden_state1=None, hidden_state2=None):
        # Process through the RNN layer
        if self.architecture == 'LSTM':
            if hidden_state1 is None or hidden_state2 is None:
                hidden_state1, hidden_state2 = self.init_hidden(x.size(0))
            output, (hidden_state1, hidden_state2) = self.rnn(x, (hidden_state1, hidden_state2))
        elif self.architecture == 'GRU':
            if hidden_state1 is None:
                hidden_state1 = self.init_hidden(x.size(0))[0]
            output, hidden_state1 = self.rnn(x, hidden_state1)
            hidden_state2 = None
        else:  # RNN
            if hidden_state1 is None:
                hidden_state1 = self.init_hidden(x.size(0))[0]
            output, hidden_state1 = self.rnn(x, hidden_state1)
            hidden_state2 = None
        
        # Process through dense layers
        x = output[:, -1, :]  # Take the last output
        for dense in self.dense_layers:
            x = torch.relu(dense(x))
            x = self.dropout(x)
        
        # Output layer
        q_values = self.output_layer(x)
        
        return q_values, hidden_state1, hidden_state2
    
    def init_hidden(self, batch_size):
        if self.architecture == 'LSTM':
            return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                    torch.zeros(1, batch_size, self.hidden_size, device=device))
        elif self.architecture == 'GRU' or self.architecture == 'RNN':
            return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                    None)
    
    def save_weights(self, is_target, ticker, dense_layers, dense_size, hidden_size, dropout_rate, input_size, num_actions):
        model_name = f"target_dqn_{ticker}" if is_target else f"dqn_{ticker}"
        model_path = f"Models/{model_name}.pth"
        torch.save(self.state_dict(), model_path)
        
    def load_weights(self, is_target, ticker, dense_layers, dense_size, hidden_size, dropout_rate, input_size, num_actions):
        model_name = f"target_dqn_{ticker}" if is_target else f"dqn_{ticker}"
        model_path = f"Models/{model_name}.pth"
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Loaded weights from {model_path}")
        except:
            print(f"No weights found at {model_path}, using random initialization")

# Replay Memory for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def save_memory(self, ticker):
        # This is a placeholder - in a real implementation you would save to disk
        print(f"Memory would be saved for {ticker}")
    
    def load_memory(self, ticker):
        # This is a placeholder - in a real implementation you would load from disk
        print(f"Memory would be loaded for {ticker}")

# Epsilon Greedy Strategy for exploration
class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * np.exp(-1. * current_step * self.decay)

# Simple Stock Environment
class StockEnvironment:
    def __init__(self, starting_cash, starting_shares, window_size, feature_size, price_column, data, scaled_data,
                 trade_fee=0.001, overtrading_penalty=0.001, drawdown_penalty=0.1):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.starting_shares = starting_shares
        self.shares = starting_shares
        self.window_size = window_size
        self.feature_size = feature_size
        self.price_column = price_column
        self.data = data  # Original price data
        self.scaled_data = scaled_data  # Normalized data with all features
        self.current_step = 0
        self.state = None
        self.time_data = None
        self.reward_history = []
        self.action_history = []
        self.portfolio_history = []
        self.q_values_history = []
        self.max_portfolio_value = starting_cash
        
            # Penalties
        self.trade_fee = trade_fee  # Transaction cost
        self.overtrading_penalty = overtrading_penalty  # Penalty for frequent trading
        self.drawdown_penalty = drawdown_penalty  # Penalty factor for drawdowns
        self.last_action = 5  # Initialize with "Hold"
        self.trade_count = 0  # Count trades for overtrading penalty
        
        
    def initialize_state(self):
        # Create initial state vector
        self.state = torch.tensor([self.scaled_data[0:self.window_size]], device=device).float()
        
    def reset(self):
        # Reset environment to initial state
        self.current_step = 0
        self.cash = self.starting_cash
        self.shares = self.starting_shares
        self.reward_history = []
        self.action_history = []
        self.portfolio_history = []
        self.q_values_history = []
        self.max_portfolio_value = self.starting_cash
        self.last_action = 5  # Hold
        self.trade_count = 0
        self.initialize_state()
        return self.state
    
    def step(self, action, q_values=None, epsilon = 0):
        # Take action and update environment
        
        if q_values is not None:
            self.q_values_history.append(q_values.max().item()) 
    
        action_value = action.item()
        self.action_history.append(action_value)
        
        # Get current price
        current_price = self.data[self.current_step + self.window_size][self.price_column]
        
        # Action mapping (for 11 actions):
        # 0: Sell 100% of shares
        # 1: Sell 80% of shares
        # 2: Sell 60% of shares
        # 3: Sell 40% of shares
        # 4: Sell 20% of shares
        # 5: Hold
        # 6: Buy shares worth 20% of cash
        # 7: Buy shares worth 40% of cash
        # 8: Buy shares worth 60% of cash
        # 9: Buy shares worth 80% of cash
        # 10: Buy shares worth 100% of cash
        
        # Calculate portfolio value before action
        portfolio_value_before = self.cash + (self.shares * current_price)
        
        self.portfolio_history.append(portfolio_value_before)
        
        # Track max portfolio value for drawdown calculation
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value_before)
        
        # Check if this is a new trade (different from last action)
        is_new_trade = (self.last_action < 5 and action_value > 5) or \
                       (self.last_action > 5 and action_value < 5) or \
                       (self.last_action != 5 and action_value == 5) or \
                       (self.last_action == 5 and action_value != 5)
        
        if is_new_trade:
            self.trade_count += 1
        
        # Execute action
        trade_cost = 0
        
        # Execute action
        if action_value < 5:  # Sell
            sell_percentage = 1.0 - (action_value * 0.2)
            shares_to_sell = int(self.shares * sell_percentage)
            self.cash += shares_to_sell * current_price
            self.shares -= shares_to_sell
        elif action_value > 5:  # Buy
            buy_percentage = (action_value - 5) * 0.2
            shares_to_buy = int((self.cash * buy_percentage) / current_price)
            self.cash -= shares_to_buy * current_price
            self.shares += shares_to_buy
        # If action is 5, hold, do nothing
        
        self.last_action = action_value
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step + self.window_size >= len(self.scaled_data) - 1
        
        # Get new state
        if not done:
            new_state = torch.tensor([self.scaled_data[self.current_step:self.current_step + self.window_size]], 
                                      device=device).float()
            # Calculate reward
            next_price = self.data[self.current_step + self.window_size][self.price_column]
            portfolio_value_after = self.cash + (self.shares * next_price)
            reward = portfolio_value_after - portfolio_value_before
              # Penalty for transaction costs (already factored into portfolio value)
            
            # Penalty for overtrading
            overtrading_penalty = 0
            if self.trade_count > 10:  # Threshold for overtrading
                overtrading_penalty = self.overtrading_penalty * reward
            
            # Penalty for drawdown
            drawdown = (self.max_portfolio_value - portfolio_value_after) / self.max_portfolio_value
            drawdown_penalty = 0
            if drawdown > 0.05:  # Only penalize significant drawdowns (>5%)
                drawdown_penalty = self.drawdown_penalty * drawdown * abs(reward)
            
            # Combine all components
            adjusted_reward = reward - overtrading_penalty - drawdown_penalty
            self.reward_history.append(adjusted_reward)
        else:
            new_state = self.state  # Just return current state if done
            adjusted_reward = 0
            
        return new_state, adjusted_reward, done
    
    def soft_reset(self, new_data, new_scaled_data, new_time_data):
        # Update data and reset state without changing cash/shares
        self.data = new_data
        self.scaled_data = new_scaled_data
        self.time_data = new_time_data
        self.current_step = 0
        self.initialize_state()
        return self.state

# Function to update Q-values
def update_Q_values(batch, Q_network, target_network, optimizer, architecture='LSTM', gamma=0.99):
    # Process batch data
    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.tensor(batch.reward, device=device)
    next_states = torch.cat(batch.next_state)
    
    # Get Q values from current network
    if architecture == 'LSTM':
        hidden_state1_batch = torch.cat([h.unsqueeze(0) for h in batch.hidden_state1])
        hidden_state2_batch = torch.cat([h.unsqueeze(0) for h in batch.hidden_state2])
        next_hidden_state1_batch = torch.cat([h.unsqueeze(0) for h in batch.next_hidden_state1])
        next_hidden_state2_batch = torch.cat([h.unsqueeze(0) for h in batch.next_hidden_state2])
        
        current_q_values, _, _ = Q_network(states, hidden_state1_batch, hidden_state2_batch)
        next_q_values, _, _ = target_network(next_states, next_hidden_state1_batch, next_hidden_state2_batch)
    else:
        current_q_values, _, _ = Q_network(states)
        next_q_values, _, _ = target_network(next_states)
    
    # Get the Q values for the actions taken
    current_q_values = current_q_values.gather(1, actions)
    
    # Compute target Q values
    max_next_q_values = next_q_values.max(1)[0]
    expected_q_values = rewards + (gamma * max_next_q_values)
    
    # Calculate loss
    loss = nn.SmoothL1Loss()(current_q_values, expected_q_values.unsqueeze(1))
    
    # Optimize model
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to avoid exploding gradients
    torch.nn.utils.clip_grad_norm_(Q_network.parameters(), 100)
    optimizer.step()
    
    
# Data loading functions
def get_real_stock_data(ticker, start_date, end_date, interval='1d'):
    """
    Get real stock data from Yahoo Finance
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        # Save the downloaded data to a CSV file
        os.makedirs('Data', exist_ok=True)  # Make sure Data/ exists
        data.to_csv(f'Data/{ticker}_{start_date}_{end_date}.csv')
        print(f"Successfully downloaded {len(data)} records for {ticker}")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def add_technical_indicators(data, ticker= None):
    """
    Add technical indicators to the dataframe
    """
    df = data.copy()
    
    # Print the columns to debug
    # print(f"Input columns: {df.columns.tolist()}")
    
    # Make sure we have the necessary columns
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        # Extract the first level column names
        first_level_cols = [col[0] for col in df.columns]
        
        # Check for required columns
        if 'Close' in first_level_cols:
            close_idx = first_level_cols.index('Close')
            close_col = df.columns[close_idx]
            close_series = df[close_col].squeeze()
        elif 'Adj Close' in first_level_cols:
            close_idx = first_level_cols.index('Adj Close')
            close_col = df.columns[close_idx]
            close_series = df[close_col].squeeze()
        else:
            print("Neither 'Close' nor 'Adj Close' found. Using first column.")
            close_series = df.iloc[:, 0].squeeze()
        
        if 'High' in first_level_cols:
            high_idx = first_level_cols.index('High')
            high_col = df.columns[high_idx]
            high_series = df[high_col].squeeze()
        else:
            high_series = close_series
            
        if 'Low' in first_level_cols:
            low_idx = first_level_cols.index('Low')
            low_col = df.columns[low_idx]
            low_series = df[low_col].squeeze()
        else:
            low_series = close_series
            
        if 'Volume' in first_level_cols:
            volume_idx = first_level_cols.index('Volume')
            volume_col = df.columns[volume_idx]
            volume_series = df[volume_col].squeeze()
        else:
            volume_series = pd.Series(np.ones_like(close_series.values))
    else:
        # Standard columns
        if 'Close' in df.columns:
            close_series = df['Close'].squeeze()
        elif 'Adj Close' in df.columns:
            close_series = df['Adj Close'].squeeze()
        else:
            close_series = df.iloc[:, 0].squeeze()
            
        high_series = df['High'].squeeze() if 'High' in df.columns else close_series
        low_series = df['Low'].squeeze() if 'Low' in df.columns else close_series
        volume_series = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(np.ones_like(close_series.values))
    
    # Add technical indicators
    # Use a separate DataFrame to store indicators to avoid modifying the original MultiIndex
    indicators_df = pd.DataFrame(index=df.index)
    
    # Trend indicators
    indicators_df['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
    indicators_df['SMA_50'] = ta.trend.sma_indicator(close_series, window=50)
    indicators_df['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
    
    # MACD
    macd = ta.trend.MACD(close_series)
    indicators_df['MACD'] = macd.macd()
    indicators_df['MACD_Signal'] = macd.macd_signal()
    indicators_df['MACD_Hist'] = macd.macd_diff()
    
    # Momentum indicators
    indicators_df['RSI'] = ta.momentum.rsi(close_series, window=14)
    stoch = ta.momentum.StochasticOscillator(high_series, low_series, close_series)
    indicators_df['Stoch_k'] = stoch.stoch()
    indicators_df['Stoch_d'] = stoch.stoch_signal()
    
    # Volatility indicators
    bollinger = ta.volatility.BollingerBands(close_series)
    indicators_df['BB_High'] = bollinger.bollinger_hband()
    indicators_df['BB_Low'] = bollinger.bollinger_lband()
    indicators_df['BB_Mid'] = bollinger.bollinger_mavg()
    indicators_df['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series)
    
    # Volume indicators
    indicators_df['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)
    
    # VWAP - For intraday data
    indicators_df['VWAP'] = (volume_series * close_series).cumsum() / volume_series.cumsum()
    
    # Price differences
    indicators_df['Close_Pct_Change'] = close_series.pct_change()
    indicators_df['High_Low_Pct'] = (high_series - low_series) / low_series * 100
    
    # Combine the original DataFrame with the indicators
    # If df has MultiIndex columns, we need to handle it differently
    if isinstance(df.columns, pd.MultiIndex):
        # Create a second level for the indicators (empty string)
        second_level = [''] * len(indicators_df.columns)
        indicators_df.columns = pd.MultiIndex.from_tuples(
            list(zip(indicators_df.columns, second_level))
        )
        # Concatenate horizontally
        result = pd.concat([df, indicators_df], axis=1)
    else:
        result = pd.concat([df, indicators_df], axis=1)
    
    # Ensure we have a 'Close' column (or equivalent) for other functions
    if isinstance(result.columns, pd.MultiIndex):
        if ('Close', '') not in result.columns and ('Close', ticker) in result.columns:
            # Copy the Close column with an empty second level
            result[('Close', '')] = result[('Close', ticker)]
    else:
        if 'Close' not in result.columns and 'Adj Close' in result.columns:
            result['Close'] = result['Adj Close']
    
    # Drop NaN values resulting from indicator calculation
    result = result.dropna()
    
    # Print the columns after adding indicators
    # print(f"Output columns: {result.columns.tolist()}")
    
    return result

def preprocess_data(data):
    """
    Preprocess and normalize the data for the model
    """
    df = data.copy()
    
    # Print the columns to debug
    # print(f"Available columns: {df.columns.tolist()}")
    
   # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        # Check if 'Close' exists in the first level of the MultiIndex
        first_level_cols = [col[0] for col in df.columns]
        
        # Find the index of 'Close' in the first level
        if 'Close' in first_level_cols:
            price_col_idx = first_level_cols.index('Close')
            print(f"Using MultiIndex 'Close' column at index {price_col_idx}")
        elif 'Adj Close' in first_level_cols:
            price_col_idx = first_level_cols.index('Adj Close')
            print(f"Using MultiIndex 'Adj Close' column at index {price_col_idx}")
        else:
            # As a last resort, use the 4th column (index 3)
            price_col_idx = 3 if len(df.columns) > 3 else 0
            print(f"No suitable price column found, using index {price_col_idx}")
        
        # Flatten the MultiIndex to make it easier to work with
        df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
    else:
        # Standard columns
        if 'Close' in df.columns:
            price_col_idx = df.columns.tolist().index('Close')
        elif 'Adj Close' in df.columns:
            price_col_idx = df.columns.tolist().index('Adj Close')
            print("Using 'Adj Close' instead of 'Close'")
        else:
            # As a last resort, use the 4th column (index 3)
            price_col_idx = 3 if len(df.columns) > 3 else 0
            print(f"No suitable price column found, using index {price_col_idx}")
    
    # Convert to numpy array
    feature_cols = df.columns.tolist()
    # print(f"Feature columns after processing: {feature_cols}")
    print(f"Price column index: {price_col_idx}")
 
    
    # Convert to numpy and handle any remaining NaN values
    features = df.values
    features = np.nan_to_num(features)
    
    # Normalize features
    scaler = features.mean(axis=0)
    scaler_nonzero = np.where(scaler == 0, 1, scaler)  # Avoid division by zero
    scaled_features = features / scaler_nonzero[None, :]
    
    # Create time features if needed (e.g., day of week, hour of day)
    time_data = np.zeros((len(df), 5))  # Placeholder
    
    return features, scaled_features, time_data, price_col_idx

# Data loading functions
def get_and_process_data(ticker, interval, window_size, start_date, end_date):
    """
    Load stock data and preprocess it for the model.
    """
    # In a real implementation, this would load data from a source
    
    data = get_real_stock_data(ticker, start_date, end_date, interval)
    
    if data is None or len(data) < window_size:
        print("Insufficient data, falling back to synthetic data")
        # Create synthetic data as fallback
        num_samples = 1000
        synthetic_data = np.random.randn(num_samples, 5)
        base_price = 150.0
        for i in range(1, num_samples):
            synthetic_data[i, 0] = synthetic_data[i-1, 3] + np.random.normal(0, 1)
            synthetic_data[i, 1] = synthetic_data[i, 0] + abs(np.random.normal(0, 1))
            synthetic_data[i, 2] = synthetic_data[i, 0] - abs(np.random.normal(0, 1))
            synthetic_data[i, 3] = synthetic_data[i, 0] + np.random.normal(0, 1)
            synthetic_data[i, 4] = abs(np.random.normal(1000000, 200000))
        
        synthetic_data[:, 0:4] += base_price
        
        # Create synthetic indicators (30 extra features)
        indicators = np.random.randn(num_samples, 30)
        
        # Combine price data and indicators
        combined_data = np.concatenate((synthetic_data, indicators), axis=1)
        price_col_idx = 3  # Close price column index
        
        # Simple scaling
        scaler = combined_data.mean(axis=0)
        scaler_nonzero = np.where(scaler == 0, 1, scaler)
        scaled_data = combined_data / scaler_nonzero[None, :]
        
        # Synthetic time data
        time_data = np.random.randn(num_samples, 5)
        
        return synthetic_data, scaled_data, time_data, price_col_idx
    
    
    # Add technical indicators
    data_with_indicators = add_technical_indicators(data)
    
    # Process and normalize data
    original_data, scaled_data, time_data, price_col_idx = preprocess_data(data_with_indicators)
    
    if len(original_data) <= window_size:
        print(f"Warning: Data length ({len(original_data)}) is less than or equal to window size ({window_size})")
        # Add padding if needed
        if len(original_data) < window_size:
            padding_size = window_size - len(original_data)
            original_data = np.pad(original_data, ((padding_size, 0), (0, 0)), 'edge')
            scaled_data = np.pad(scaled_data, ((padding_size, 0), (0, 0)), 'edge')
            time_data = np.pad(time_data, ((padding_size, 0), (0, 0)), 'edge')
    
    return original_data, scaled_data, time_data, price_col_idx

def get_all_months(start_date, end_date):
    """
    Generate a list of date strings from start date to end date.
    """
    # Use 'MS' for month start or 'M' for month end
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    date_strings = [date.strftime('%Y-%m-%d') for date in dates]
    return date_strings


def calculate_metrics(portfolio_history, risk_free_rate=0.02):
    """
    Calculate performance metrics from portfolio history
    """
    if not portfolio_history:
        return {
            'Total Return': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown': 0,
            'Win/Loss Ratio': 0,
            'Annualized Volatility': 0
        }
    
    # Convert to numpy array for calculations
    values = np.array(portfolio_history)
    
    # Calculate daily returns
    returns = np.diff(values) / values[:-1]
    
    # Total return
    total_return = (values[-1] / values[0]) - 1
    
    # Sharpe ratio (annualized)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (mean_return - risk_free_rate/252) / std_return * np.sqrt(252)
    
    # Maximum drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Win/Loss ratio
    wins = len(returns[returns > 0])
    losses = len(returns[returns < 0])
    win_loss_ratio = wins / losses if losses > 0 else float('inf')
    
    # Annualized volatility
    annualized_volatility = std_return * np.sqrt(252)
    
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win/Loss Ratio': win_loss_ratio,
        'Annualized Volatility': annualized_volatility
    }

def moving_average_crossover_strategy(data, short_window=20, long_window=50):
    """
    Simple MA crossover strategy for comparison
    """
    df = data.copy()
    
    
     # Determine which column to use for price
    if 'Close' in df.columns:
        price_col = 'Close'
    elif 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif len(df.columns) >= 4:
        # Assume 4th column is close price
        price_col = df.columns[3]
    else:
        # Use the first column as a fallback
        price_col = df.columns[0]
        
        
     # Calculate moving averages
    df['SMA_Short'] = df[price_col].rolling(window=short_window).mean()
    df['SMA_Long'] = df[price_col].rolling(window=long_window).mean()
    
    # Generate signals (1: Buy, 0: Hold, -1: Sell)
    df['Signal'] = 0
    df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
    df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1
    
    # Calculate strategy returns
    df['Returns'] = df[price_col].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    
    # Calculate cumulative returns
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Strategy_Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    
    return df


def visualize_results(env, ticker, start_date, end_date, price_data):
    """
    Create visualizations of trading results
    """
    # Create figures directory if it doesn't exist
    os.makedirs('Results/Figures', exist_ok=True)
    
    # Extract data for plotting
    actions = env.action_history
    rewards = env.reward_history
    portfolio_values = env.portfolio_history
    q_values = env.q_values_history if hasattr(env, 'q_values_history') else []
    
    # Convert to numpy arrays for easier manipulation
    actions = np.array(actions)
    
    # Create a dataframe with the price data
    if isinstance(price_data, pd.DataFrame):
        prices = price_data['Close'].values
    else:
        # If price_data is a numpy array, assume Close is the 4th column (index 3)
        prices = price_data[:, 3]
    
    dates = pd.date_range(start=start_date, periods=len(prices), freq='D')
    if len(dates) > len(prices):
        dates = dates[:len(prices)]
    
    # 1. Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title(f'Equity Curve - {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig(f'Results/Figures/{ticker}_equity_curve.png')
    
    # 2. Trading Signals on Price Chart
    plt.figure(figsize=(12, 8))
    
    # Only plot signals where we have action data
    max_len = min(len(prices), len(actions))
    plot_prices = prices[:max_len]
    plot_actions = actions[:max_len]
    
    plt.plot(plot_prices, label='Price', alpha=0.7)
    
    # Find buy and sell signals
    buy_indices = np.where(plot_actions > 5)[0]  # Buy actions are > 5
    sell_indices = np.where(plot_actions < 5)[0]  # Sell actions are < 5
    hold_indices = np.where(plot_actions == 5)[0]  # Hold actions are == 5
    
    if len(buy_indices) > 0:
        plt.scatter(buy_indices, plot_prices[buy_indices], color='green', marker='^', s=100, label='Buy')
    if len(sell_indices) > 0:
        plt.scatter(sell_indices, plot_prices[sell_indices], color='red', marker='v', s=100, label='Sell')
    if len(hold_indices) > 0 and len(hold_indices) < 100:  # Only show holds if there aren't too many
        plt.scatter(hold_indices, plot_prices[hold_indices], color='blue', marker='o', s=50, alpha=0.5, label='Hold')
    
    plt.title(f'Price Chart with Trading Signals - {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Results/Figures/{ticker}_trading_signals.png')
    
    # 3. Q-Values Over Time (if available)
    if q_values:
        plt.figure(figsize=(12, 6))
        plt.plot(q_values)
        plt.title(f'Q-Values Over Time - {ticker}')
        plt.xlabel('Time Steps')
        plt.ylabel('Max Q-Value')
        plt.grid(True)
        plt.savefig(f'Results/Figures/{ticker}_q_values.png')
    
    # 4. Rewards Distribution
    if rewards:
        plt.figure(figsize=(12, 6))
        plt.hist(rewards, bins=50)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'Reward Distribution - {ticker}')
        plt.xlabel('Reward Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'Results/Figures/{ticker}_rewards_dist.png')
    
    # 5. Action Distribution
    plt.figure(figsize=(12, 6))
    unique_actions, action_counts = np.unique(actions, return_counts=True)
    plt.bar([str(int(a)) for a in unique_actions], action_counts)
    plt.title(f'Action Distribution - {ticker}')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig(f'Results/Figures/{ticker}_action_dist.png')
    
    plt.close('all')
    print(f"Visualizations saved to Results/Figures/")

def compare_strategies(env, price_data, ticker, start_date, end_date):
    """
    Compare DQN strategy with baseline strategies
    """
    # Convert price data to DataFrame if it's not already
    if not isinstance(price_data, pd.DataFrame):
        if len(price_data.shape) > 1:
            # Assuming OHLCV format
            df = pd.DataFrame(price_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        else:
            df = pd.DataFrame(price_data, columns=['Close'])
    else:
        df = price_data.copy()
    

def initialize(feature_size=22):
    """
    Initialize environment, DQN networks, optimizer and memory replay.
    """
    architecture = "RNN"
    starting_cash = 10000
    starting_shares = 0
    window_size = 128
    lookback_period = 400
    price_column = 3
    dense_size = 128
    dense_layers = 2
    feature_size = 22
    hidden_size = 128
    num_actions = 11
    dropout_rate = 0.2

    env = StockEnvironment(starting_cash=starting_cash, starting_shares=starting_shares, window_size=window_size,
                           feature_size=feature_size, price_column=price_column, data=[], scaled_data=[])

    memoryReplay = ReplayMemory(100000)

    Q_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
                    architecture=architecture, dense_layers=dense_layers, dense_size=dense_size,
                    dropout_rate=dropout_rate)

    target_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
                         architecture=architecture, dense_layers=dense_layers, dense_size=dense_size,
                         dropout_rate=dropout_rate)

    target_network.load_state_dict(Q_network.state_dict())
    optimizer = optim.Adam(Q_network.parameters(), lr=0.0001)

    hidden_state1, hidden_state2 = Q_network.init_hidden(1)

    return env, memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2

def execute_action(state, hidden_state1, hidden_state2, epsilon, num_actions, Q_network):
    sample = random()
    q_values = None
    
    if sample > epsilon:
        with torch.no_grad():
            q_values, hidden_state1, hidden_state2 = Q_network(state, hidden_state1, hidden_state2)
            action = q_values.max(1)[1].view(1, 1)
            return action, hidden_state1, hidden_state2, q_values, epsilon
    else:
        action = torch.tensor([[randrange(num_actions)]], device=device, dtype=torch.long)
        return action, hidden_state1, hidden_state2, q_values, epsilon

def main_loop(ticker, start_date, end_date, window_size=128, C=5, BATCH_SIZE=512, architecture='RNN', train_test_split=0.8):
    """
    Run the main loop of DQN training.
    """
    # Set the interval, threshold, years, and months for data retrieval
    interval = '1d'
    
    all_months = get_all_months(start_date, end_date)
    
    # Split into training and testing periods
    split_idx = int(len(all_months) * train_test_split)
    training_months = all_months[:split_idx]
    testing_months = all_months[split_idx:]
    
    print(f"Training on {len(training_months)} months, testing on {len(testing_months)} months")
    
    # Initialize the environment, memory replay, Q-network, target network, optimizer, and hidden states
 # Get data first to determine feature size
    test_data, test_scaled_data, test_time_data, _ = get_and_process_data(ticker, interval, window_size, start_date, end_date)
    
    # Determine feature size dynamically
    feature_size = test_scaled_data.shape[1]
    print(f"Detected feature size: {feature_size}")
    
    # Then initialize with the correct feature size
    env, memoryReplay, num_actions, Q_network, target_network, optimizer, last_hidden_state, current_hidden_state = initialize(feature_size=feature_size)

    Q_network.load_weights(False, ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size, Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)
    
    target_network.load_weights(True, ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size, Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)
    
    memoryReplay.load_memory(ticker)
    
    Q_network = Q_network.to(device)
    target_network = target_network.to(device)
    
    prev_hidden_state_1 = None
    prev_hidden_state_2 = None
    
    steps_done = 0
    iterator = 0
    reward = 0
    
    epsilon_strategy = EpsilonGreedyStrategy(start=1.00, end=0.01, decay=0.000001)

# TRAINING PHASE
    print("Starting TRAINING phase...")
    # Loop through all months
    for month in training_months:
        iterator += 1
        print(f"Processing training month {month} ({iterator}/{len(training_months)})")
        
        # Retrieve and process data for the current month
        if iterator == 1:
            env.data, env.scaled_data, env.time_data, scaler = get_and_process_data(ticker, interval, window_size,start_date,end_date)
            env.initialize_state()
            state = env.reset()
            prev_hidden_state_1, prev_hidden_state_2 = Q_network.init_hidden(1)
        else:
            new_data, new_scaled_data, time_data, scaler = get_and_process_data(ticker, interval, window_size,start_date,end_date)
            print("RESET")
            state = env.soft_reset(new_data, new_scaled_data, time_data)

        done = False
        episode_reward = 0
        
        # Loop until the episode is done
        while not done:
            steps_done += 1

            epsilon = epsilon_strategy.get_exploration_rate(steps_done)

            # Execute an action and get the next state, reward, and done flag
            action, curr_hidden_state1, curr_hidden_state2, q_values, epsilon = execute_action(
                state, last_hidden_state, current_hidden_state, epsilon, num_actions, Q_network)
            
            next_state, reward_delta, done = env.step(action, q_values, epsilon)

            episode_reward += reward_delta
            reward += reward_delta

            if prev_hidden_state_1 is not None and prev_hidden_state_2 is not None:
                # Add the transition to the memory replay
                transition = Transition(
                    state=state, 
                    hidden_state1=prev_hidden_state_1, 
                    hidden_state2=prev_hidden_state_2,
                    action=action, 
                    next_state=next_state, 
                    reward=reward_delta, 
                    next_hidden_state1=curr_hidden_state1,
                    next_hidden_state2=curr_hidden_state2
                )
                memoryReplay.push(transition)

            # Update previous hidden states
            prev_hidden_state_1 = curr_hidden_state1
            prev_hidden_state_2 = curr_hidden_state2
            
            # If the memory replay is full, sample a batch and update the Q-values every 4 steps
            if len(memoryReplay) >= BATCH_SIZE and steps_done % 4 == 0:
                transitions = memoryReplay.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                update_Q_values(batch, Q_network, target_network, optimizer, architecture)

            # If the number of steps is a multiple of C (100), update the target network
            if steps_done % 100 == 0:
                target_network.load_state_dict(Q_network.state_dict())
                print(f"Step {steps_done}: Updated target network, current reward: {reward}")

            state = next_state
            
        print(f"Month {month} completed. Episode reward: {episode_reward}")

        # Save weights after each month
        Q_network.save_weights(False, ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size,
                               Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

        target_network.save_weights(True, ticker, Q_network.dense_layers_num, Q_network.dense_size,
                                    Q_network.hidden_size, Q_network.dropout_rate, Q_network.input_size, 
                                    Q_network.num_actions)

        memoryReplay.save_memory(ticker)

    print(f"Training completed. Final reward: {reward}")
    
   
     # TESTING PHASE
    print("Starting TESTING phase...")
    test_portfolio_history = []
    original_cash = env.starting_cash
    original_shares = env.starting_shares
    
    # Reset environment for testing
    env.cash = original_cash
    env.shares = original_shares
    
    for month in testing_months:
        print(f"Testing on month {month}")
        
        # Get data for the test month
        test_data, test_scaled_data, test_time_data, _ = get_and_process_data(ticker, interval, window_size, start_date,end_date)
        test_state = env.soft_reset(test_data, test_scaled_data, test_time_data)
        
        done = False
        test_rewards = []
        test_actions = []
        
        # Initialize hidden states for testing
        test_hidden_state1, test_hidden_state2 = Q_network.init_hidden(1)
        
        # Evaluate without exploration (epsilon=0)
        while not done:
            # Use Q_network for inference
            with torch.no_grad():
                q_values, test_hidden_state1, test_hidden_state2 = Q_network(test_state, test_hidden_state1, test_hidden_state2)
                action = q_values.max(1)[1].view(1, 1)
            
            # Take step in environment
            next_state, reward, done = env.step(action, q_values, 0)
            
            test_rewards.append(reward)
            test_actions.append(action.item())
            test_state = next_state
        
        test_portfolio_history.extend(env.portfolio_history)
        
        # Calculate performance metrics for this test month
        metrics = calculate_metrics(env.portfolio_history)
        print(f"Test metrics for {month}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Calculate overall test performance
    overall_metrics = calculate_metrics(test_portfolio_history)
    print("Overall test performance:")
    for key, value in overall_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Compare with baseline strategies
    if isinstance(test_data, np.ndarray):
    # If test_data has more columns than our basic list, we need to create extended column names
        if test_data.shape[1] > 5:
        # Create basic columns for the first 5 (if they exist)
            base_columns = ['Open', 'High', 'Low', 'Close', 'Volume'][:min(5,     test_data.shape[1])]
        
            # Add generic feature names for the remaining columns
            extra_columns = [f'Feature_{i}' for i in range(len(base_columns),     test_data.shape[1])]
            
            # Combine all column names
            all_columns = base_columns + extra_columns
            
            # Create DataFrame with all columns
            test_df = pd.DataFrame(test_data, columns=all_columns)
        else:
            # If test_data has 5 or fewer columns, use standard column names
            cols = ['Open', 'High', 'Low', 'Close', 'Volume'][:test_data.shape[1]]
            test_df = pd.DataFrame(test_data, columns=cols)
    else:
        test_df = test_data.copy()
    
    # Run baseline strategy (Moving Average Crossover)
    ma_results = moving_average_crossover_strategy(test_df)
    
    # Calculate baseline metrics
    baseline_returns = ma_results['Strategy_Cumulative_Returns'].dropna().values
    baseline_metrics = {
        'Total Return': baseline_returns[-1] / baseline_returns[0] - 1 if len(baseline_returns) > 0 else 0,
        'Sharpe Ratio': ma_results['Strategy_Returns'].mean() / ma_results['Strategy_Returns'].std() * np.sqrt(252) if len(ma_results) > 0 else 0,
        # Other metrics can be calculated similarly
    }
    
    print("Baseline strategy performance:")
    for key, value in baseline_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize results
    visualize_results(env, ticker, start_date, end_date, test_data)
    
    print(f"Testing completed. Final portfolio value: {env.portfolio_history[-1]:.2f}")

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2022-03-31"
    
    print("Starting DQN trading agent training...")
    print(f"Ticker: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    
    main_loop(ticker=ticker, start_date=start_date, end_date=end_date)