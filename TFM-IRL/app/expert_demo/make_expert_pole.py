import gym
import keyboard
import numpy as np

# MACROS
Push_Left = 0
Push_Right = 1

# Key mapping
arrow_keys = {
    'a': Push_Left,
    'd': Push_Right}

env = gym.make('CartPole-v1')


trajectories = []
episode_step = 0

for episode in range(20): # n_trajectories : 20
    trajectory = []
    step = 0

    env.reset()
    print("episode_step", episode_step)

    while True: 
        env.render()
        print("step", step)

        key = keyboard.read_key()
        if key not in arrow_keys.keys():
           break
        action = arrow_keys[key]
        
        state, reward, done, _ = env.step(action)

        if done==True: # trajectory_length : 130
           break

        trajectory.append((state[0], state[1], action))
        step += 1
    trajectory_numpy = np.array(trajectory)
    print("trajectory_numpy.shape", trajectory_numpy.shape)
    episode_step += 1
    trajectories.append(trajectory)

np_trajectories = np.array(trajectories, dtype=object)
print("np_trajectories.shape", np_trajectories.shape)

np.save("app\expert_demo\expert_demo_pole.npy", arr=np_trajectories)