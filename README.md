## Bipedal Walker OpenAI gym Reinforcement Learning solution.

![bipedal walker solution](/bipedal_walker_solution/ending.gif)

___


## Resources:
___

### Codebases:
- [philtabor](https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG/tensorflow/walker2d/ddpg_orig_tf.py)
- [keiohta](https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/ddpg.py)
- [stevenpjg](https://github.com/stevenpjg/ddpg-aigym/issues/7)
- [keras docs](https://keras.io/examples/rl/ddpg_pendulum/)

### Papers:
- [Understanding Failures In Deterministic Environments With Sparse Rewards](https://arxiv.org/pdf/1911.11679.pdf)

### Posts:
- [Practical tips](https://www.reddit.com/r/reinforcementlearning/comments/7s8px9/deep_reinforcement_learning_practical_tips/)


### Lessons:

List of stupid mistakes made throughout implementing this algorithm.

1. Always check numpy array shapes. Specifically that you haven't broadcast a (64) dimension array over a (64, 1) dimension array! 🤦
2. Check every variable. Spent ages trying to figure out why nothing was being learnt only to discover instead of returning states and next_states from the memory buffer sample I was instead just returning states and states! 🤦
3. Copied and pasted the actor network while building the critic and accidentally forgot to remove the `tanh` activation meaning the critic could at most predict a total of reward `1` or `-1` for the entire episode given any state and action pair! 🤦
4. Left the hard-coded high action bound in from training the pendulum environment as a default when initializing the actor model. Correctly adjusted it for the actor on the agent class but not the target actor meaning the target actor would always output 2 times the action the actor would! 🤦
