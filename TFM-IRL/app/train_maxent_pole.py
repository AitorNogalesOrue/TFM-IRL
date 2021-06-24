import sys
sys.path.append('./')
import gym
import pylab
import numpy as np

import maxent

n_states = 24*24 # position - 20, velocity - 20
n_actions = 2
one_feature = 24 # number of state per one feature
q_table = np.zeros((n_states, n_actions)) # (400, 3)
feature_matrix = np.eye((n_states)) # (400, 400)

gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05

np.random.seed(1)

def idx_demo(env, one_feature):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance_0 = (env_high[0]*10 - env_low[0]*10)/one_feature
    env_distance_2 = (env_high[2]*10 - env_low[2]*10)/one_feature

    raw_demo = np.load(file="app\expert_demo\expert_demo_pole.npy")
    demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))

    for x in range(len(raw_demo)):
        for y in range(len(raw_demo[0])):
            position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance_0)
            angel_idx = int((raw_demo[x][y][2] - env_low[2]) / env_distance_2)
            state_idx = position_idx + angel_idx * one_feature

            demonstrations[x][y][0] = state_idx
            demonstrations[x][y][1] = raw_demo[x][y][2] 
            
    return demonstrations

def idx_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    s=state*10
    env_distance_0 = (env_high[0]*10 - env_low[0]*10)/one_feature
    env_distance_2 = (env_high[2]*10 - env_low[2]*10)/one_feature
    
    position=int((s[0]-env_low[0]*10)/env_distance_0)
    angel=int((s[2]-env_low[2]*10)/env_distance_2)
    state_idx = position + angel 
    return state_idx

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)


def main():
    env = gym.make('CartPole-v1')
    demonstrations = idx_demo(env, one_feature)

    expert = maxent.expert_feature_expectations(feature_matrix, demonstrations)
    learner_feature_expectations = np.zeros(n_states)

    theta = -(np.random.uniform(size=(n_states,)))

    episodes, scores = [], []

    for episode in range(30000):
        state = env.reset()
        score = 0

        if (episode != 0 and episode == 10000) or (episode > 10000 and episode % 5000 == 0):
            learner = learner_feature_expectations / episode
            maxent.maxent_irl(expert, learner, theta, theta_learning_rate)
                
        while True:
            state_idx = idx_state(env, state)
            action = np.argmax(q_table[state_idx])
            
            next_state, reward, done, _ = env.step(action)
            
            irl_reward = maxent.get_reward(feature_matrix, theta, n_states, state_idx)
            next_state_idx = idx_state(env, next_state)
            update_q_table(state_idx, action, irl_reward, next_state_idx)
            
            learner_feature_expectations += feature_matrix[int(state_idx)]

            score += reward
            state = next_state
            
            if done:
                scores.append(score)
                episodes.append(episode)
                break

        if episode % 1000 == 0:
            score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("app/learning_curves/maxent_pole_30000.png")
            np.save("app/results/maxent_q_table_pole", arr=q_table)

if __name__ == '__main__':
    main()