import gym
import keyboard
import numpy as np

# MACROS
Do_Nothing = 0
Fire_Left_Engine = 1
Fire_Down_Engine = 2
Fire_Rignt_Engine = 3

# Key mapping
arrow_keys = {
    's': Do_Nothing,
    'd': Fire_Left_Engine,
    'w': Fire_Down_Engine,
    'a': Fire_Rignt_Engine}

env = gym.make('LunarLander-v2')


trajectories = []
episode_step = 0
print(env.action_space)
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

np.save("gail\expert_demo\expert_demo_pole.npy", arr=np_trajectories)