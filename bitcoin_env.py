import gym
from gym import spaces
import numpy as np

class BitcoinTradingEnv(gym.Env):
    def __init__(self, initial_balance=1000000):
        super(BitcoinTradingEnv, self).__init__()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.bitcoin = 0
        self.current_price = 50000  # 초기 비트코인 가격

        # Action Space: (action_type, percentage)
        # action_type: 0=매도, 1=매수
        # percentage: 1~10 (10% ~ 100%)
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),  # 0 또는 1
            spaces.Discrete(10)  # 0~9 (1=10%, 10=100%)
        ))

        # Observation Space
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )

    def reset(self):
        self.balance = self.initial_balance
        self.bitcoin = 0
        self.current_price = 50000
        return np.array([self.balance, self.bitcoin, self.current_price], dtype=np.float32)

    def step(self, action):
        action_type, percentage_index = action
        trade_percentage = (percentage_index + 1) / 10  # 10% ~ 100%
        
        if action_type == 1:  # 매수
            trade_amount = trade_percentage * self.balance
            if self.balance >= trade_amount:  # 자산 확인
                self.bitcoin += trade_amount / self.current_price
                self.balance -= trade_amount
        elif action_type == 0:  # 매도
            trade_amount = trade_percentage * self.bitcoin * self.current_price
            if self.bitcoin * self.current_price >= trade_amount:  # 보유량 확인
                self.balance += trade_amount
                self.bitcoin -= trade_amount / self.current_price

        # 가격 변동 (단순 랜덤)
        self.current_price *= np.random.uniform(0.95, 1.05)

        # 상태 업데이트 및 보상 계산
        total_assets = self.balance + self.bitcoin * self.current_price
        reward = total_assets - self.initial_balance  # 자산 변화량
        done = False  # 에피소드 종료 조건 (필요시 구현)
        obs = np.array([self.balance, self.bitcoin, self.current_price], dtype=np.float32)

        return obs, reward, done, {}

    def render(self, mode="human"):
        print(f"Balance: {self.balance}, Bitcoin: {self.bitcoin}, Price: {self.current_price}")
