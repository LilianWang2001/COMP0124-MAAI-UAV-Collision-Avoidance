# COMP0124-MAAI-UAV-Collision-Avoidance

## MAPPO (based on [OpenRL](https://github.com/OpenRL-Lab/openrl))

The training and evaluation process of [MAPPO](https://arxiv.org/abs/2103.01955) are implemented in ```openrl``` folder.

To enable the OpenRL library
```
cd openrl
pip install .
```

Our custom MPE for MAPPO algorithm is defined in ```openrl/openrl/envs/mpe/scenarios/simple_spread.py```.

The trained MAPPO model is stored in ```openrl/examples/mpe/ppo_agent``` as .pt file.

To visualize the result of the trained MAPPO in our environment
```
cd openrl/examples/mpe/
python3 test_ppo.py
```

The gif file ```ppo.gif``` recorded the whole environment and actions of agents derived from trained MAPPO model.

When keeping the current setting in ```simple_spread.py```, a fixed environment will be generated as follow
<div align="center">
  <img src="openrl/examples/mpe/ppo_fixed_env.gif" height="200" width="200"></a>
</div>

By changing the initial fixed states in environment to the random sampled case, the result will be
<div align="center">
  <img src="openrl/examples/mpe/ppo.gif" height="500" width="500"></a>
</div>

## QMIX (based on [MARL Benchmark](https://github.com/OpenRL-Lab/openrl))

The training and evaluation process of [MAPPO](https://arxiv.org/abs/2103.01955) are implemented in ```openrl``` folder.

To enable the off-policy library
```
# install off-policy package
cd off-policy
pip install -e .
```
