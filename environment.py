import gym
from gym import spaces
import numpy as np

class MultiStockTradingEnv(gym.Env):
    def __init__(self, stock_data):
        super(MultiStockTradingEnv, self).__init__()
        self.stock_data = stock_data  # {'A': [가격 리스트], 'B': [가격 리스트]}
        self.current_step = 0
        self.balance = 10000  # 초기 현금
        
        # 상태 공간: 종목별 현재 가격과 잔액을 포함
        self.observation_space = spaces.Dict({
            "price_A": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "price_B": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

        # 행동 공간: 0=아무것도 안 함, 1=매수 A, 2=매도 A, 3=매수 B, 4=매도 B
        self.action_space = spaces.Discrete(5)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        return self._next_observation()

    def _next_observation(self):
        price_A = self.stock_data['A'][self.current_step]
        price_B = self.stock_data['B'][self.current_step]
        return {
            "price_A": np.array([price_A]),
            "price_B": np.array([price_B]),
            "balance": np.array([self.balance])
        }

    def step(self, action):
        price_A = self.stock_data['A'][self.current_step]
        price_B = self.stock_data['B'][self.current_step]
        
        if action == 1:  # 매수 A
            shares_bought = self.balance // price_A
            self.balance -= shares_bought * price_A
        elif action == 2:  # 매도 A
            self.balance += shares_bought * price_A
        elif action == 3:  # 매수 B
            shares_bought = self.balance // price_B
            self.balance -= shares_bought * price_B
        elif action == 4:  # 매도 B
            self.balance += shares_bought * price_B
        
        # 보상 계산 (총 자산 변화율)
        new_total_assets = self.balance
        reward = new_total_assets - self.balance

        self.current_step += 1
        done = self.current_step >= len(self.stock_data['A']) - 1

        return self._next_observation(), reward, done, {}

    def render(self, mode="human"):
        price_A = self.stock_data['A'][self.current_step]
        price_B = self.stock_data['B'][self.current_step]
        print(f"Step: {self.current_step}, Price A: {price_A}, Price B: {price_B}, Balance: {self.balance}")
