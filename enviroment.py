def initialize_env(env, obs_dim = 1):
    observation, info = env.reset()
    if obs_dim == 1:
        return [observation], info
    else:
        return observation, info

def run_env_step(env, action, random_action=False, obs_dim = 1):
    if random_action :
        action = env.action_space.sample()  # agent policy that uses the observation and info (we just sample a random action )
    observation, reward, terminated, truncated, info = env.step(action)
    if obs_dim == 1:
        return [observation], reward, terminated, truncated, info
    else:
        return observation, reward, terminated, truncated, info

def env_reset(env):
    observation, info = env.reset()
    return observation, info

def close_env(env):
    env.close()

def custom_reward(state, reward, terminated, game = "Frozen"):
    if game == "Frozen":
            if not terminated:
                reward = 0
            elif terminated and reward == 1:
                reward = 10
            else:
                reward = 0
    else:
        pass
    return reward