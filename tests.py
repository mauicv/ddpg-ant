from src.memory import ReplayBuffer
from src.agent import Agent
from src.train import Train
import numpy as np


def test_replay_buffer_size_constraints():
    action_space_dim = 2
    state_space_dim = 4
    replay_buffer = ReplayBuffer(
        action_space_dim=action_space_dim,
        state_space_dim=state_space_dim,
        size=3)

    actions = []
    states = []
    rewards = []
    dones = []

    for i in range(4):
        action = np.random.rand((action_space_dim))
        state = np.random.rand((state_space_dim))
        next_state = np.random.rand((state_space_dim))
        reward = np.random.rand((1))
        done = np.random.rand((1))

        replay_buffer.push(state, next_state, action, reward, done)
        actions.append(action)
        states.append(state)
        rewards.append(reward)
        dones.append(done)

    assert(np.all(actions[3] == replay_buffer.actions[0]))
    assert(np.all(actions[1] == replay_buffer.actions[1]))
    assert(np.all(actions[0] != replay_buffer.actions[0]))

    assert(np.all(states[3] == replay_buffer.states[0]))
    assert(np.all(states[1] == replay_buffer.states[1]))
    assert(np.all(states[0] != replay_buffer.states[0]))

    assert(np.all(rewards[3] == replay_buffer.rewards[0]))
    assert(np.all(rewards[1] == replay_buffer.rewards[1]))
    assert(np.all(rewards[0] != replay_buffer.rewards[0]))

    assert(len(replay_buffer.dones) == 3)


def test_replay_buffer_sample():
    action_space_dim = 2
    state_space_dim = 4
    sample_size = 5
    replay_buffer = ReplayBuffer(
        action_space_dim=action_space_dim,
        state_space_dim=state_space_dim,
        size=10,
        sample_size=sample_size)

    for i in range(20):
        action = np.random.rand((action_space_dim))
        state = np.random.rand((state_space_dim))
        next_state = np.random.rand((state_space_dim))
        reward = np.random.rand((1))
        done = np.random.rand((1))
        replay_buffer.push(state, next_state, action, reward, done)

    states, next_states, actions, rewards, dones = replay_buffer.sample()
    assert(states.shape == (5, state_space_dim))
    assert(next_states.shape == (5, state_space_dim))
    assert(actions.shape == (5, action_space_dim))
    assert(rewards.shape == (5,))
    assert(dones.shape == (5,))


def test_agent():
    state_space_dim = 3
    action_space_dim = 4
    train = Train()
    agent = Agent(state_space_dim=state_space_dim,
                  action_space_dim=action_space_dim,
                  low_action=-1,
                  high_action=1,
                  load=False)
    state = np.random.rand((state_space_dim))[None]
    next_state = np.random.rand((state_space_dim))[None]
    action = agent.get_action(state)
    reward = np.array([1])
    done = np.array([0])
    Q_loss, policy_loss = train(agent, state, next_state, action, reward, done)
    assert(True)


tests = [
    test_replay_buffer_size_constraints,
    test_replay_buffer_sample,
    test_agent
]

for test in tests:
    test()
