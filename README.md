# Rainbow DQN
(Almost) Rainbow DQN implementation in Pytorch for [Gymnasium](https://gymnasium.farama.org/)'s Lunar Lander.

Implementation follows [Rainbow paper (Hessel 2017)](https://arxiv.org/abs/1710.02298) by combining following improvements on basic Deep Q-Network algorithm:
1. Double Q-Learning - [(van Hasselt 2015)](https://arxiv.org/pdf/1509.06461.pdf)
2. N-step learning - [(Sutton 1988)](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
3. Noisy linear layers - [(Fortunato et al. 2017)](https://arxiv.org/pdf/1706.10295.pdf)
4. Dueling network architecture - [(Wang et al. 2015)](https://arxiv.org/pdf/1511.06581.pdf)
5. Prioritized experience replay - [(Schaul et al. 2015)](https://arxiv.org/pdf/1511.05952.pdf)

Notably, this is not a full Rainbow, as Distributional reinforcement learning from [paper by Bellemere et al. 2017](https://arxiv.org/abs/1707.06887) is not implemented. This was due to computational restrictions. 

## Results
