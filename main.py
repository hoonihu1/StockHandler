from bitcoin_env import BitcoinTradingEnv
import DQNAgent

env = BitcoinTradingEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space[1].n  # percentage index

agent = DQNAgent(state_dim, action_dim)
episodes = 500
batch_size = 32

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action_index = agent.act(state)
        action_type = action_index // 10  # 0=매도, 1=매수
        percentage_index = action_index % 10
        action = (action_type, percentage_index)

        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action_index, reward, next_state, done)

        state = next_state
        total_reward += reward

        agent.replay(batch_size)

    agent.update_target_model()
    print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}")
