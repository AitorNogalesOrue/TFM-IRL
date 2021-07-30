import sys
sys.path.append('./')
import gym
import pylab
import numpy as np

import app_pole

n_states = 24*24 # position - 20, velocity - 20
n_actions = 2
one_feature = 24 # number of state per one feature
feature_num = 4
q_table = np.zeros((n_states, n_actions))  # (400, 3)

gamma = 0.99
q_learning_rate = 0.03

#def idx_state(env, state):
#    env_low = env.observation_space.low
#    env_high = env.observation_space.high
#    s=state*10
#    env_distance_0 = (env_high[0]*10 - env_low[0]*10)
#    env_distance_2 = (env_high[2]*10 - env_low[2]*10)
    
#    position=int((s[0]+env_low[0]*10)/env_distance_0)
#    angel=int((s[2]+env_low[2]*10)/env_distance_2)
#    state_idx = position + angel
#    return state_idx

def idx_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    s=state*100
    position=int(s[0])
    angel=int(s[2])
    state_idx = position + angel

    return state_idx


def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)


def main():
    env = gym.make('CartPole-v1')
    demonstrations = np.load(file="expert_demo\expert_demo_pole.npy")
    
    feature_estimate = app_pole.FeatureEstimate(feature_num, env)
    
    learner = app_pole.calc_feature_expectation(feature_num, gamma, q_table, demonstrations, env)
    learner = np.matrix([learner])
    
    expert = app_pole.expert_feature_expectation(feature_num, gamma, demonstrations, env)
    expert = np.matrix([expert])
    
    w, status = app_pole.QP_optimizer(feature_num, learner, expert)
    
    
    episodes, scores = [], []
    
    for episode in range(60000):
        state = env.reset()
        score = 0

        while True:
            state_idx = idx_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, reward, done, _ = env.step(action)
            
            features = feature_estimate.get_features(state)
            irl_reward = np.dot(w, features)
            
            next_state_idx = idx_state(env, next_state)
            update_q_table(state_idx, action, irl_reward, next_state_idx)

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
            pylab.savefig("learning_curves/app_eps_pole_60000.png")
            np.save("results/app_q_table_pole.npy", arr=q_table)

        if episode % 5000 == 0:
            # optimize weight per 5000 episode
            status = "infeasible"
            temp_learner = app_pole.calc_feature_expectation(feature_num, gamma, q_table, demonstrations, env)
            learner = app_pole.add_feature_expectation(learner, temp_learner)
            
            while status=="infeasible":
                w, status = app_pole.QP_optimizer(feature_num, learner, expert)
                if status=="infeasible":
                    learner = app_pole.subtract_feature_expectation(learner)

if __name__ == '__main__':
    main()