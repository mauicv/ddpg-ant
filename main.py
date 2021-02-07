import pybullet_envs
import gym
import click
import os
import shutil
from time import time
import numpy as np

from src.agent import Agent
from src.memory import ReplayBuffer
from src.train import Train
from src.logging import Logging
from src.noise import OUNoise, NormalNoise # noqa


# ENV_NAME = 'Pendulum-v0'
ENV_NAME = 'AntBulletEnv-v0'
LAYERS_DIMS = [400, 300]
TAU = 0.001
SIGMA = 0.15*40
THETA = 0.2*40
BUFFER_SIZE = 100000
BATCH_SIZE = 64
DISCOUNT = 0.99
ACTOR_LR = 0.00005
CRITIC_LR = 0.0005


def setup_env():
    env = gym.make(ENV_NAME)
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    state_norm_array = env.observation_space.high
    min_action = env.action_space.low.min()
    max_action = env.action_space.high.max()
    if np.any(np.isinf(state_norm_array)):
        state_norm_array = np.ones_like(state_norm_array)
    return env, state_space_dim, action_space_dim, state_norm_array, \
        min_action, max_action


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        pass


@cli.command()
@click.pass_context
@click.option('--episodes', '-e', default=5000, type=int,
              help='Number of epsiodes of training')
@click.option('--steps', '-s', default=400, type=int,
              help='Max number of steps per episode')
def train(ctx, episodes, steps):
    logger = Logging([
        'episode',
        'rewards',
        'running_40_episode_reward',
        'episode_length',
        'epsiode_run_time',
        'average_step_run_time',
        'q_loss',
        'p_loss'
    ])

    env, state_space_dim, action_space_dim, state_norm_array, min_action, \
        max_action = setup_env()

    replay_buffer = ReplayBuffer(state_space_dim=state_space_dim,
                                 action_space_dim=action_space_dim,
                                 size=BUFFER_SIZE,
                                 sample_size=BATCH_SIZE)

    noise_process = OUNoise(
        dim=action_space_dim,
        sigma=SIGMA,
        theta=THETA,
        dt=1e-2)

    # noise_process = NormalNoise(
    #     dim=action_space_dim,
    #     sigma=SIGMA)

    agent = Agent(state_space_dim,
                  action_space_dim,
                  layer_dims=LAYERS_DIMS,
                  low_action=min_action,
                  high_action=max_action,
                  noise_process=noise_process,
                  tau=TAU,
                  load=True)

    train = Train(discount_factor=DISCOUNT,
                  actor_learning_rate=ACTOR_LR,
                  critic_learning_rate=CRITIC_LR)

    training_rewards = []
    for episode in range(episodes):
        noise_process.reset()
        state = np.array(env.reset(), dtype='float32')
        episode_reward = 0
        step_count = 0
        done = False
        episode_start_time = time()
        step_times = []
        q_losses = []
        p_losses = []
        while not done:
            if step_count >= steps:
                break

            step_time_start = time()
            step_count += 1

            # environment step
            action = agent.get_action(state[None], with_exploration=True)[0]
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push((state, next_state, action, reward, done))
            state = next_state

            # training step
            if replay_buffer.ready:
                states, next_states, actions, \
                    rewards, dones = replay_buffer.sample()
                q_loss, p_loss = \
                    train(agent, states, next_states,
                          actions, rewards, dones)
                agent.track_weights()

            if replay_buffer.ready:
                q_losses.append(q_loss.numpy())
                p_losses.append(p_loss.numpy())
            episode_reward += reward
            step_time_end = time()
            step_times.append(step_time_end - step_time_start)
        training_rewards.append(episode_reward)
        episode_end_time = time()
        epsiode_time = episode_end_time - episode_start_time
        average_step_time = np.array(step_times).mean()
        average_q_loss = np.array(q_losses).mean()
        average_p_loss = np.array(p_losses).mean()
        running_40_episode_reward = np.mean(training_rewards[-40:])

        logger.log([episode, episode_reward, running_40_episode_reward,
                    step_count, epsiode_time, average_step_time,
                    average_q_loss, average_p_loss])

        agent.save_models()


@cli.command()
@click.pass_context
@click.option('--steps', '-s', default=400, type=int,
              help='Max number of steps per episode')
@click.option('--noise', '-n', is_flag=True,
              help='With exploration')
def play(ctx, steps, noise):
    env, state_space_dim, action_space_dim, state_norm_array, min_action, \
        max_action = setup_env()

    noise_process = OUNoise(
        dim=action_space_dim,
        sigma=SIGMA,
        theta=THETA,
        dt=1e-2)

    # noise_process = NormalNoise(
    #     dim=action_space_dim,
    #     sigma=SIGMA)

    agent = Agent(state_space_dim,
                  action_space_dim,
                  layer_dims=LAYERS_DIMS,
                  low_action=min_action,
                  high_action=max_action,
                  noise_process=noise_process,
                  load=True)
    env.render()
    state = env.reset()

    agent.actor.summary()
    agent.critic.summary()
    r = 0
    for i in range(steps):
        action = agent.get_action(state[None], with_exploration=noise)[0]
        state, reward, done, _ = env \
            .step(action)
        print(reward)
        r = r + reward
        state = state
        env.render()
    print(r)

@cli.command()
@click.pass_context
def clean(ctx):
    for save_path in os.listdir('save'):
        path = f'./save/{save_path}'
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


if __name__ == '__main__':
    cli()
