# main.py
# 导入环境和学习方法
from .env import ArmEnv
from .rl import DDPG

# 设置全局变量
MAX_EPISODES = 500
MAX_EP_STEPS = 200

# 设置环境
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# 设置学习方法 (这里使用 DDPG)
rl = DDPG(a_dim, s_dim, a_bound)

# 开始训练
for i in range(MAX_EPISODES):
    # 初始化回合设置
    s = env.reset()

    for j in range(MAX_EP_STEPS):
        # 环境的渲染
        env.render()
        # RL 选择动作
        a = rl.choose_action(s)
        # 选取的动作与环境交互，得到 observation, reward, done
        s_, r, done = env.step(a)

        # DDPG 这种强化学习需要存放记忆库
        rl.store_transition(s, a, r, s_)

        if rl.memory_full:
            # 记忆库满了, 开始学习
            rl.learn()

        # 变为下一回合
        s = s_