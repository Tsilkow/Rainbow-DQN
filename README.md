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
  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/plots/agent_314159.png">
  <br>
  Plot of evaluation score against simulation steps
  <br>
  <br>
  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_20000.gif">
  <br>
  Evaluation after 20000 steps of training
  <br>
  <br>
  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_40000.gif">
  <br>
  Evaluation after 40000 steps of training
  <br>
  <br>
  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_60000.gif">
  <br>
  Evaluation after 60000 steps of training
  <br>
  <br>
  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_80000.gif">
  <br>
  Evaluation after 80000 steps of training
  <br>
  <br>
  <img src="https://raw.githubusercontent.com/Tsilkow/Rainbow-DQN/main/video/agent_314159_100000.gif">
  <br>
  Evaluation after 100000 steps of training
</p>

## How to run
**Pytorch does not yet support Python 3.11, so it's recommended to use Python 3.10**

In order to achieve different training and evaluation results, it is important to change seed in Hyperparameters class. Feel free to change other values, to experiment!

Before using the script, make sure to install the dependencies:
```bash
pip install -r requirements.txt
```
  
To train new agent, simply run `python main.py`, if you want to record some evaluations, add flag `-r`. Note: recording is mutually exclusive with presenting the evaulation on screen due to the way gymnasium is set up.  

To load a specific agent, include flag `-l` and specify file with network data from 'agents/'.
