'''
Gym 的一个最小例子 CartPole-v0

env.step() 返回:
    observation: 执行当前step后，环境的观测
    reward: 执行上一步 action 后， agent 获得的奖励
    done: boolen 值，表示是否需要将环境重制 env.reset()。大多数情况下，Done为true时，表示当前 episode/trial结束
    info: 针对调试过程的诊断信息

    该函数在仿真器中扮演物理引擎的角色。其输入是动作 action，在该函数中，
    一般利用智能体的运动学模型和动力学模型计算下一步的状态和立即回报，并判断是否达到终止状态。

在 gym 仿真中，每一次回合开始，需要先执行 reset() 函数，返回初始观测信息，然后根据标志位 done 的状态，来决定是否进行下一次 episode。
    remarks:
        在强化学习算法中，agent需要一次次地尝试，累积经验，然后从经验中学到好的动作。
        一次尝试我们称之为trial或episode. 每次trial都要到达终止状态. 一次trial结束后，agent需要从头开始，
        这就需要agent具有重新初始化的功能。函数reset()就是这个作用。
'''

import gym


class MyGymDemo:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def firstDemo(self):
        """
        选取随机动作 demo
        """
        for i_episode in range(20):
            # 20 个 episode，每次都要得到初始观测信息，env.rest()
            observation = self.env.reset()

            # print('当前初始状态：', observation)

            for t in range(100):
                self.env.render()
                print('循环内状态: ', observation)

                # 每次随机选取动作
                action = self.env.action_space.sample()
                # 与环境交互，获得下一步
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

    def useSpace(self):
        """
        在 gym 中，有运动空间 action_space 和 观测空间 observation_space 两个指标
        程序中被定义为 Space 类型，用于描述有效的运动和观测的格式和范围

        action_space: 是一个 discrete 类型，{0,1,..., n-1} 长度为 n 的非负整数集合，在 'CartPole-v0' 例子中，动作空间为{0,1},即左和右
        observation_space: 是一个 box 类型，表示一个 n 维的盒子
            remarks: A (possibly unbounded) box in R^n. Specifically, a Box represents the
                    Cartesian product of n closed intervals. Each interval has the form of one
                    of [a, b], (-oo, b], [a, oo), or (-oo, oo).
        """
        print(self.env.action_space)
        print(self.env.observation_space)


if __name__ == '__main__':
    # 选定 example
    env_name = 'CartPole-v0'

    MyGymDemo(env_name).useSpace()

