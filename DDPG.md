# DDPG 算法
DDPG 全称 Deep Deterministic Policy Gradient，
本质上是**AC**框架的一种强化学习算法，结合了基于policy的policy Gradient 和
基于action value的 DQN，可以通过off-policy的方法，单步更新 policy，
预测出确定性策略，进而实现 total reward最大化。

DDPG 的出现，使得连续动作的直接预测问题成为可能。


## 参考
[强化学习：DDPG算法详解及调参记录](https://zhuanlan.zhihu.com/p/84321382)
[一文带你理清DDPG算法（附代码及代码解释）](https://zhuanlan.zhihu.com/p/111257402)
