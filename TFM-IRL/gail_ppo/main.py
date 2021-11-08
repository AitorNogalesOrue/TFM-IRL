import os
import gym
import argparse
import numpy as np
from collections import deque
import pickle
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from ppo import train_model
from gail import train_actor_critic, train_discrim
from model import Actor, Critic, Discriminator
from utils.zfilter import ZFilter
from utils.utils import *

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--gail', type=bool, default=True,
                    help='choose model gail or ppo') 
parser.add_argument('--env_name', type=str, default="LunarLanderContinuous-v2", 
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None, 
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False, 
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=64, 
                    help='hidden unit size of actor, critic networks (default: 64)')
parser.add_argument('--learning_rate', type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--discrim_update_num', type=int, default=2, 
                    help='update number of discriminator (default: 2)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--model_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                    help='GAIL update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=2048, 
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
                    help='GAIL accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
                    help='GAIL accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--episode_num', type=int, default=10000,
                    help='maximal number of episodes (default: 250)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

def main():
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)

    print('state size:', num_inputs) 
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions, args.hidden_size)
    critic = Critic(num_inputs, args.hidden_size)
    
    if args.gail:
        discrim = Discriminator(num_inputs + num_actions, args.hidden_size)

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate, 
                              weight_decay=args.l2_rate)
    if args.gail:
        discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)
        expert_demo = pickle.load(open('expert_demo\expert_demo_300_ms244.665498101087.p', "rb"))
        demonstrations = np.array(expert_demo)
        print("demonstrations.shape", demonstrations.shape)
        writer = SummaryWriter(comment="-gail_episode_num-epi_" + str(args.episode_num)+"-shape_"+str(demonstrations.shape))
    else:
        writer = SummaryWriter(comment="-ppo_iter-epi_" + str(args.episode_num))
    
    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        if args.gail:
            discrim.load_state_dict(ckpt['discrim'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    
    episodes = 0    
    iter = 0
    train_discrim_flag = True
    while episodes <= args.episode_num:
        iter+=1
        #actor.eval(), critic.eval()
        memory = deque()
        steps = 0
        scores = []

        while steps < args.sample: 
            state = env.reset()
            score = 0

            state = running_state(state)
            
            for _ in range(10000): 
                if args.render:
                    env.render()

                steps += 1

                mu, std = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)
                if args.gail:
                    irl_reward = get_reward(discrim, state, action)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                next_state = running_state(next_state)
                state = next_state

                score += reward

                if done:
                    break
            
            episodes += 1
            scores.append(score)

        score_avg = np.mean(scores)
        print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), episodes)
        if args.gail:
            actor.train(), critic.train(), discrim.train()
            if train_discrim_flag:
                expert_acc, learner_acc = train_discrim(discrim, memory, discrim_optim, demonstrations, args)
                print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
                if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                    train_discrim_flag = False
            train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)
        else:
            actor.train(), critic.train()
            train_model(actor, critic, memory, actor_optim, critic_optim, args)

        if iter % 1000:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            
            if args.gail:
                gail_ckpt_path = os.path.join(model_path, 'gail_ckpt_'+ str(score_avg)+'.pth.tar')

                save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'discrim': discrim.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=gail_ckpt_path)
            else:
                ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')
                save_checkpoint({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'z_filter_n':running_state.rs.n,
                    'z_filter_m': running_state.rs.mean,
                    'z_filter_s': running_state.rs.sum_square,
                    'args': args,
                    'score': score_avg
                }, filename=ckpt_path)

if __name__=="__main__":
    main()