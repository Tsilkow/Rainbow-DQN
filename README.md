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
Following results were achieved by training on my personal computer without GPU (but I'm hoping to get to train a more ambitious example on a GPU) for 100000 steps.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/plots/agent_314159.png" width=50% height=auto>

  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_20000.gif" width=50% height=auto>
  <br>
  Evaluation after 20000 steps of training
  <br>

  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_40000.gif" width=50% height=auto>
  <br>
  Evaluation after 40000 steps of training
  <br>

  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_60000.gif" width=50% height=auto>
  <br>
  Evaluation after 60000 steps of training
  <br>

  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_80000.gif" width=50% height=auto>
  <br>
  Evaluation after 80000 steps of training
  <br>

  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_100000.gif" width=50% height=auto>
  <br>
  Evaluation after 100000 steps of training
</p>
