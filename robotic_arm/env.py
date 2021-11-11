import pyglet
import numpy as np

class ArmEnv(object):
    viewer = None
    # 转动一步的时间
    dt = 0.1
    # 转动的角度范围，每一步能转动的范围
    action_bound = [-1, 1]
    # goal 的 x,y 坐标和长度 l
    goal = {'x': 100., 'y': 100., 'l': 40}
    # 两个观测值（观测两个关节的转动角）
    state_dim = 2
    # 两个动作(demo中只有两个关节可动)
    action_dim = 2

    def __init__(self):
        """
        定义手臂，保存每一节的转动角和手臂长度
        """
        self.viewer = None

        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])

        # 两段手臂都 100 长
        self.arm_info['l'] = 100
        # 两段手臂的端点角度
        self.arm_info['r'] = np.pi / 6

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        if self.viewer is None:
            # 绘制 手臂 与 目标
            self.viewer = Viewer(self.arm_info, self.goal)

        # 使用 Viewer 中的 render 功能
        self.viewer.render()


class Viewer(pyglet.window.Window):
    """
    可视化模块
    只在 env.render() 中调用
    """

    # 手臂厚度
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # 画出手臂等
        # 创建窗口的继承
        # vsync 如果是 True, 按屏幕频率刷新, 反之不按那个频率
        super(Viewer, self).__init__(
            width=400, height=400, resizable=False, caption='Arm', vsync=False)

        # 窗口背景颜色
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = np.array([200, 200])

        # 将手臂的作图信息放入 batch
        # 每次渲染都是将 batch 中的内容一次性展示出来
        self.batch = pyglet.graphics.Batch()

        # 矩形, 有四个顶点, 就能使用 GL_QUADS 这种形式
        self.point = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color

        # 添加两条手臂
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))

        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        """刷新并呈现在屏幕上"""
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        """刷新手臂等位置"""

        # 清屏
        self.clear()

        # 画上 batch 里面的内容
        self.batch.draw()

    def _update_arm(self):
        # 更新手臂的位置信息
        pass


if __name__ == '__main__':
    arm_info = np.zeros(
        2, dtype=[('l', np.float32), ('r', np.float32)])
    print(arm_info)
    # 生成出 (2,2) 的矩阵
    # 两段手臂都 100 长
    arm_info['l'] = 100
    # 两段手臂的端点角度
    arm_info['r'] = np.pi / 6
    print(arm_info)