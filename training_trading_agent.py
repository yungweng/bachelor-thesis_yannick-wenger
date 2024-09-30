import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces
from keras import layers
import time
from keras import initializers

# Changeable Variables
initial_balance = 1000000  # Starting balance of $1 million
max_trade_fraction = 0.99  # Maximum 100% of balance per trade
episodes = 10000  # Number of episodes
gamma = 0.2  # Discount factor for reward
batch_size = 20  # Batch size for training
patience = 20 # Patience for Early Stopping

transaction_cost = 0 # Transaction Cost of total volume

# Load and preprocess stock data
df = pd.read_csv('AAPL.csv') # Load Apple stock data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.set_index('Date')

# Calculate 14-period RSI
delta = df['Close'].diff()
up = delta.where(delta > 0, 0)
down = -delta.where(delta < 0, 0)
rs = up.rolling(window=14).mean() / down.rolling(window=14).mean()
df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

# Calculate 12 and 26-period EMA for MACD
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Calculate 14-period RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

# Calculate 20-period CCI
tp = (df['High'] + df['Low'] + df['Close']) / 3
sma_tp = tp.rolling(window=20).mean()
mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
df['CCI'] = (tp - sma_tp) / (0.015 * mean_dev)

# Calculate 14-period ADX
high_diff = df['High'].diff()
low_diff = df['Low'].diff()
df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
tr = pd.concat(
    [df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1))],
    axis=1).max(axis=1)
atr = tr.ewm(span=14, adjust=False).mean()
df['+DI'] = 100 * (df['+DM'].ewm(span=14, adjust=False).mean() / atr)
df['-DI'] = 100 * (df['-DM'].ewm(span=14, adjust=False).mean() / atr)
dx = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
df['ADX'] = dx.ewm(span=14, adjust=False).mean()

df.dropna(inplace=True) # Remove NaN values

df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'RSI', 'ADX', 'CCI']] # Remove unwanted columns in data
# Split data into training and testing sets
train_data = df['2010-01-01':'2016-12-31'] # Training dataset

# Creation of stock trading environment
class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.reward = None
        self.reward_range = (-np.inf, np.inf)

        self.max_portfolio_value = None
        self.agent_value = None
        self.previous_agent_value = None
        self.buy_and_hold_value = None

        self.losing_streak = None

        self.total_shares_sold = None
        self.total_shares_bought = None

        self.df = df

        self.action_space = spaces.Box(
            low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.df.columns),), dtype=np.float32
        )

        self.initial_price = self.df['Close'].iloc[0]
        self.buy_and_hold_shares = initial_balance // self.initial_price

        self.reset()

    def reset(self):
        self.reward = 0

        self.balance = initial_balance
        self.agent_value = initial_balance
        self.previous_agent_value = initial_balance
        self.max_portfolio_value = initial_balance
        self.buy_and_hold_value = initial_balance

        self.current_step = 0

        self.long_shares_held = 0
        self.total_shares_sold = 0
        self.total_shares_bought = 0

        self.losing_streak = 0
        self.buy_and_hold_shares = self.buy_and_hold_shares

        return self._next_observation()

    def _next_observation(self):
        # Observe data from dataset
        data = self.df.iloc[self.current_step].values

        # Observe data from agent
        agent = np.array([
            self.balance,
            self.agent_value,
            self.long_shares_held,
            self.buy_and_hold_shares,
        ])

        # Concatenate data
        observation = np.concatenate((data, agent))

        return observation

    def step(self, action):
        # Reset reward for current step
        self.reward = 0
        # take action for current step
        self._take_action(action)
        self.current_step += 1
        # Check if training is finished
        done = self.current_step >= len(self.df) - 1
        # Load next obsverations
        obs = self._next_observation()

        # Set current price
        current_price = self.df.iloc[self.current_step]['Close']
        # Set previous price
        previous_price = self.df.iloc[self.current_step - 1]['Close']

        # Exit program if agent has a balance < 0
        self.agent_value = self.balance + (self.long_shares_held * current_price)
        if self.balance < 0:
            print(f'Balance < 0 with {self.balance}')
            exit()

        # Exit program if agent has a value < 0
        if self.agent_value < 0:
            print(f'Agent Value < 0 with {self.agent_value}')
            exit()

        # Update buy and hold value for comparison
        self.buy_and_hold_value = self.buy_and_hold_shares * current_price
        self.max_portfolio_value = max(self.max_portfolio_value, self.agent_value)

        # Set reward to difference between agent value and buy and hold value
        self.reward += (self.agent_value - self.buy_and_hold_value)

        if previous_price < current_price:
            self.reward += self.long_shares_held * (current_price - previous_price)
        else:
            self.reward += self.long_shares_held * (current_price - previous_price) * 3



        print(f'####### Difference {self.agent_value - self.buy_and_hold_value} #######')
        print(f'Reward {self.reward}')

        self.render(self)
        return obs, self.reward, done, {}

    def _take_action(self, action):
        # Set current price for action
        current_price = self.df.iloc[self.current_step]['Close']

        # Action type > 0 = buy, Action type < 0 = sell stock
        action_type = action[0]
        # amount of possible shares bought / sold
        amount = action[1]

        if action_type > 0:  # Buy
            total_cost_per_share = current_price * (1 + transaction_cost)
            max_trade_amount = self.balance * max_trade_fraction
            total_possible = int(max_trade_amount / total_cost_per_share)
            shares_bought = int(total_possible * amount)
            if shares_bought > 0:  # Buy
                self.balance -= shares_bought * current_price
                self.balance -= transaction_cost * shares_bought * current_price
                self.long_shares_held += shares_bought
                self.total_shares_bought += shares_bought
                print(f'Agent bought {shares_bought} shares at a price of ${current_price} each for a total of ${shares_bought * current_price * (1 + transaction_cost)}')

        elif action_type < 0:  # Sell
            shares_sold = int(self.long_shares_held * amount)
            if shares_sold > 0:

                self.balance += shares_sold * current_price
                self.balance -= transaction_cost * shares_sold * current_price
                self.long_shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                print(f'Agent sold {shares_sold} shares at a price of ${current_price} each for a total of ${shares_sold * current_price * (1 + transaction_cost)}')

    def render(self, mode='human', close=False):
        profit = self.agent_value - initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Agent Value: {self.agent_value}')
        print(f'Buy and Hold Value: {self.buy_and_hold_value}')
        print(f'Long Shares held: {self.long_shares_held}')
        print(f'Profit: {profit}')


class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()
        self.clip_epsilon = 0.2

        # Create learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0008,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def build_actor(self):
        model = tf.keras.models.Sequential([
            layers.Input(shape=(self.state_size + 4,)),
            layers.Dense(1024, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(1024, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(1024, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(512, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(512, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(256, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(256, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(128, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(128, kernel_initializer=initializers.HeUniform()),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.GaussianNoise(0.1),
            layers.Dense(self.action_size, activation='tanh')
        ])
        return model

    def build_critic(self):
        model = tf.keras.models.Sequential([
            layers.Input(shape=(self.state_size + 4,)),
            layers.Dense(256, kernel_initializer=initializers.HeUniform()),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(128, kernel_initializer=initializers.HeUniform()),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(64, kernel_initializer=initializers.HeUniform()),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(32, kernel_initializer=initializers.HeUniform()),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Dense(1)
        ])
        return model

    def policy(self, state):
        state = np.expand_dims(state, axis=0)
        action_mean = self.actor_model(state)
        action_std = 0.4
        action_mean = np.squeeze(action_mean)
        action = np.random.normal(action_mean, action_std)
        action = np.clip(action, [-1, 0], [1, 1])
        log_prob = -0.5 * np.sum(((action - action_mean) / action_std) ** 2 + np.log(2 * np.pi * action_std ** 2))
        return action, log_prob

    def update(self, states, actions, advantages, old_log_probs, returns):
        with tf.GradientTape() as tape:
            new_log_probs = self.actor_model(states, training=True)
            values = self.critic_model(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - values))

            # Ensure old_log_probs has the same shape as new_log_probs and correct dtype
            old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
            old_log_probs = tf.expand_dims(old_log_probs, axis=-1)

            ratio = tf.exp(new_log_probs - old_log_probs)
            # Expand dimensions of advantages to match the shape of ratio
            if len(advantages.shape) == 1:
                advantages = tf.expand_dims(advantages, axis=-1)
            advantages = tf.cast(advantages, dtype=tf.float32)  # Cast advantages to float32
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            total_loss = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(total_loss, self.actor_model.trainable_variables + self.critic_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads, self.actor_model.trainable_variables + self.critic_model.trainable_variables))


def calculate_advantages_and_returns(rewards, gamma):
    returns = []
    discounted_sum = 0
    for r in reversed(rewards):
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = np.array(returns)
    advantages = returns - np.mean(returns)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
    return advantages, returns


# Train function
def train(env, agent, episodes=episodes, batch_size=batch_size, gamma=gamma, patience=patience):
    episode_counter = 0
    patience_counter = 0
    best_reward = -np.inf
    for episode in range(episodes):
        episode_counter += 1
        print(f'####### Episode: {episode} #######')
        state = env.reset()
        done = False
        total_reward = 0
        states, actions, rewards, log_probs = [], [], [], []

        while not done:
            action, log_prob = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            total_reward += reward

        advantages, returns = calculate_advantages_and_returns(rewards, gamma)
        agent.update(np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), advantages,
                     np.array(log_probs), returns)

        # Early Stopping
        if total_reward > best_reward:
            print('####### Better model found! #######')
            best_reward = total_reward
            patience_counter = 0
            print(f'Episode {episode}, Total Reward: {best_reward}')
            print(f'Best Reward: {best_reward}')
            print('Save best Actor and Critic')
            # Save best model
            agent.actor_model.save_weights('best_actor_model.weights.h5')
            agent.critic_model.save_weights('best_critic_model.weights.h5')
            time.sleep(1)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at episode {episode}. Best reward: {best_reward}')
            print('Load best Actor and Critic')
            agent.actor_model.load_weights('best_actor_model.weights.h5')
            agent.critic_model.load_weights('best_critic_model.weights.h5')
            break

    return agent

# Test function
def test_agent(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    portfolio_values = [initial_balance]
    buy_hold_values = [initial_balance]

    while not done:
        action, _ = agent.policy(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        current_price = env.df.iloc[env.current_step]['Close']
        portfolio_value = env.balance + (env.long_shares_held * current_price)

        portfolio_values.append(portfolio_value)
        buy_hold_values.append(env.buy_and_hold_value)

    return portfolio_values, buy_hold_values, total_reward


# Main execution
print(f'####### TRAIN #######')
# Load training data
train_env = StockTradingEnv(train_data)

state_size = train_env.observation_space.shape[0]
action_size = 2  # Buy or Sell // Amount
agent = PPOAgent(state_size, action_size)
trained_agent = train(train_env, agent)