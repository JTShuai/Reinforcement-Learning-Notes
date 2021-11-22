# 强化学习
强化学习入门。

强化学习的中心思想，就是让agent在环境里学习。每个行动会对应各自的奖励(reward)，智能体通过分析数据来学习，怎样的情况下应该做怎样的事情。

强化学习学习的是一个策略，目前主要有三大类学习的框架，它们分别是:
- 策略迭代方法（policy iteration method）
- 策略梯度方法（policy gradient method）
- 无导数优化方法（derivative-free optimization method）

## 基本概念
强化学习的基本模型就是个体-环境的交互。**个体/智能体（agent）**就是能够*采取一系列行动并且期望获得较高收益或者达到某一目标*的部分，而与此相关的另外的部分我们都统一称作**环境（environment）**。

在每个时刻环境和个体都会产生相应的交互。个体可以采取一定的**行动（action）**，这样的行动是施加在环境中的。环境在接受到个体的行动之后，会反馈给个体环境目前的**状态（state）**以及由于上一个行动而产生的**奖励（reward）**。


上面所描述的 agent-env 相互作用可以使用下面的示意图表示。存在一连串的时刻 $t = 1,2,3...$，在每一个时刻中，*agent*都会接受到*env*的一个*state*信号 $S_t \in \mathcal{S}$ ，在每一步中*agent*会从该*state*允许的行动集中挑选一个来采取*action* $A_t \in \mathcal{A}(S_t)$，*env*接受到这个action信号之后，在*下一个时刻*env 会反馈给agent相应的state信号 $S_{t+1} \in \mathcal{S^{+}}$ 和即时奖励 $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$ 。其中，我们把**所有的状态**记做 $\mathcal{S^{+}}$ ，把**非终止状态**记做 $\mathcal{S}$ 。
![个体与环境相互作用](/images/2021-11-12-12-14-20.png)

强化学习的目标是希望个体从环境中获得的**总奖励最大**，即我们的目标不是短期的某一步行动之后获得最大的奖励，而是希望长期地获得更多的奖励。

### 片段性与连续性任务
强化学习里的任务分两种。在很多常见的任务中，比如下围棋，在一局棋未结束的时候奖励常常都是为零的，而仅当棋局结束的那一个时刻才会根据个体的输赢产生一个奖励值；而在另外一些任务中，环境给予奖励可能分布在几乎每个时刻中。

对于像下围棋这样存在一个*终止状态*，并且所有的 reward 会在这个终止状态及其之前结算清的任务，我们称之为**片段性任务（episodic task）**。
- 这类任务，有个*起点*，有个*终点*。两者之间有一堆状态，一堆行动，一堆奖励，和一堆新的状态，它们共同构成了一“集”。
- 当一集结束，也就是**到达终止状态**的时候，智能体会看一下奖励累积了多少，以此评估自己的表现。然后，它就带着之前的经验开始一局**新游戏**。这一次，智能体做决定的依据会充分一些。

还存在另外的一类任务，它们并不存在一个终止状态，即原则上它们可以永久地运行下去，这类任务的奖励是分散地分布在这个连续的一连串的时刻中的，我们称这一类任务为**连续性任务（continuing task）**。
- 这类任务永远不会有游戏结束的时候。智能体要学习如何选择最佳的行动，和环境进行实时交互。就像自动驾驶汽车，并没有过关拔旗子的事。
- 这样的任务是通过时间差分学习 (Temporal Difference Learning) 来训练的。每一个时间步，都会有总结学习，等不到一集结束再分析结果。

由于我们的目标是希望获得的总奖励最大，因此我们希望量化地定义这个总奖励，这里我们称之为**收益**（return）。对于*片段性任务*而言，我们可以很直接定义收益为


$G_t = R_{t+1} + R_{t+2} + ... + R_T $

其中 $T$ 为回合结束的时刻，即 $S_T$ 属于终止状态。对于**连续任务**而言，不存在一个这样的终止状态，因此，这样的定义可能会在连续任务中发散。因此我们引入另外一种收益的计算方式，称之为**衰减收益（discounted return）**:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+ ... = \sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1} \tag{1}$$

其中**衰减率（discount factor）** $\gamma$ 满足 $0 \le \gamma \le 1$ 。这样的定义也很好理解，相比于更远的收益，我们会*更加偏好临近的收益*，因此对于离得较近的收益权重更高。

### 探索与开发

探索 (Exploration) 是找到**关于环境的更多信息**。
开发 (Exploitation) 是利用已知信息来**得到最多的奖励**。
要记住，目标是将预期累积奖励最大化。正因如此，它有时候会陷入一种困境。

![探索与开发](/images/2021-11-12-12-14-20.png)

小老鼠可以吃到无穷多块分散的奶酪 (每块+1) 。但在迷宫上方，有许多堆在一起的奶酪(+1000) ，或者看成巨型奶酪。如果我们只关心吃了多少，小老鼠就永远不会去找那些大奶酪。它只会在安全的地方一块一块地吃，这样奖励累积比较慢，但它不在乎。如果它跑去远的地方，也许就会发现大奖的存在，但也有可能发生危险。

因此需要设定一种规则，让 agent 能够把握二者之间的平衡。

## 强化学习的三种方法
三种方法分别是：
-  基于策略（policy-based）：策略是决定个体行为的机制。是**从状态到行为**的一个映射，可以是确定性的，也可以是不确定性的。
- 基于价值（value-based）：是一个未来奖励的预测，用来评价当前状态的好坏程度。当面对两个不同的状态时，个体可以用一个Value值来评估这两个状态可能获得的最终奖励区别，继而指导选择不同的行为，即制定不同的策略。同时，**一个价值函数是基于某一个特定策略的，不同的策略下同一状态的价值并不相同**。
- 基于模型（model-based）：个体对环境的一个建模，它体现了个体是如何思考环境运行机制的（how the agent think what the environment was.），**个体希望模型能模拟环境与个体的交互机制**。

强化学习中的个体可以由以上三个组成部分中的一个或多个组成。

### 基于价值 Value-based
这种方法，目标是优化价值函数**V(s)**。价值函数会告诉我们，agent 在每个状态里得出的未来奖励最大预期 (maximum expected future reward) 。

注意，下一个 state $s'$只与当前的 state $s$ 和采取的 action $a$ 有关，与$s$之前的轨迹无关。且有定义可知 $V_\pi (T) = 0$，因为终止状态的未来不会有奖励。

一个状态下的**函数值**，是从当前状态开始算，agent可以预期的未来奖励积累总值。

**状态价值函数 state value function** $V_{\pi}(s)$ 表示当到达某个状态 $s$ 之后，*如果接下来一直按着策略 $\pi$ 来行动*，能够获得的期望收益,结合公式$(1)$可得：

$$V_{\pi}(s) = \mathbb{E}_{\pi}[G_t| S_t = s] = \mathbb{E}[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}|S_t=s] \tag{2.a}$$

**行动价值函数 action value function** $Q_{\pi}(s, a)$ 表示当到达某个状态 $s$ 之后，*如果采取行动 $a$，接下来再按照策略 $\pi$ 来行动*，能获得的期望收益。

$$Q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t| S_t = s, A_t = a] = \mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}|S_t=s, A_t = a] \tag{2.b}$$

#### Bellman方程推导
由公式$(1)$ 可得 $G_{t+1} = R_{t+2} + \gamma R_{t+3}+... = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1+1}$

而在公式 $(2)$ 中，$\sum_{t=0}^{\infty}\gamma^t R_{t+1} = R_{t+1} + \sum_{t=1}^{\infty}\gamma^k R_{t+k+1} = R_{t+1} + \sum_{t=0}^{\infty}\gamma^{k+1} R_{t+k+1+1}$，结合上面的$G_{t+1}$表达式可得:
$$\sum_{k=0}^{\infty}\gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1} \tag{3}$$

结合 $(2)$、$(3)$ 可得:
<!-- $$V_{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1}|S_t=s] \tag{4}$$ -->

$$
\begin{equation}
\begin{split} 
V_{\pi}(s) &= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1}|S_t=s] \\ 
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma V_{\pi}(S_{t+1})|S_t=s]\\ 
\end{split}\end{equation}
\tag{4.a}
$$

$$
Q_{\pi}(s, a) = \mathbb{E}_{\pi}[R_{t+1} + \gamma Q_{\pi}(S_{t+1}, A_{t+1}) |S_t=s, A_t = a]
\tag{4.b}
$$

通过方程可以看出 $V_{\pi}(s)$ 由两部分组成，一是该状态的**即时奖励期望**，即时奖励期望等于**即时奖励**，因为根据即时奖励的定义，它*与下一个状态无关*；另一个是**下一时刻状态的价值期望**，可以根据下一时刻状态的概率分布得到其期望。如果用 $s’$ 表示$s$状态下一时刻任一可能的状态，那么Bellman方程可以写成：
$$ V_{\pi}(s) = \mathcal{R_s} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}} V_{\pi}(s') \tag{5}$$

将 Bellman 方程转换为矩阵形式：

$$
\begin{bmatrix}
V(1) \\
\vdots\\
V(n)\\
\end{bmatrix} = 
\begin{bmatrix}
R_1 \\
\vdots\\
R_n\\
\end{bmatrix} + \gamma \begin{bmatrix}
P_{11}&\cdots&P_{1n} \\
\vdots&\ddots\\
P_{n1}&\cdots&P_{nn}\\
\end{bmatrix} \begin{bmatrix}
V(1) \\
\vdots\\
V(n)\\
\end{bmatrix}
\tag{6}
$$

上述方程可写为:

$$
\begin{equation}\begin{split} 
\boldsymbol{V} &= \boldsymbol{R} + \gamma \boldsymbol{P} \boldsymbol{V}\\
(\boldsymbol{I} - \gamma \boldsymbol{P}) \boldsymbol{V} &= \boldsymbol{R}\\ 
\boldsymbol{V} &=(\boldsymbol{I} - \gamma \boldsymbol{P})^{-1} \boldsymbol{R}\\ 
\end{split}\end{equation}
\tag{7}
$$

Bellman方程是一个线性方程组，因此理论上是可以直接求解。然后其计算复杂度为 $\mathcal{O}(n^{3})$，其中 $n$ 为状态数量，因此直接求解仅适用于小规模问题。大规模求解通常使用迭代法，如动态规划、蒙特卡洛评估、时序差分学习等。

#### 决定过程
对于决定过程，需要考虑行为集合 $\mathcal{A}$。在上文讨论的奖励过程，没有将状态转移时采取的 action 考虑进去。对于状态转移时采取 action 的数学描述为:

$$
\mathcal{P_{ss'}^{a}} = P[\mathcal{S_{t+1}} = s'｜ \mathcal{S_t} = s, \mathcal{A_t} = a] \tag{8}
$$

$$
\mathcal{R_s^a} = \mathbb{E}[R_{t+1}| \mathcal{S_t} = s, \mathcal{A_t} = a]
\tag{9}
$$
其中 $a$ 为采取的 action 。

#### 基于策略(policy)的价值函数
在上文中标记的 $\pi$ 表示采取的策略，**策略**是概率的集合或分布，策略将 *行为* 与 *状态*
关联起来，表示在某一状态 $s$ 采取可能的行为 $a$ 的概率:
$$ \pi(a|s) = P(A_t =a | S_t = s)  \tag{10}$$

> 注意式子 $(10)$ 与 $(8)、(9)$ 的区别，之前的两个公式是定义已经处于 s 状态下并且采取 行动 a 之后，得到下一状态 $s'$ 的概率，而式$(10)$ 表示的是，处于状态 $s$ 时，采取行动 $a$ 的概率。

Policy 仅与**当前的状态**有关, 与历史信息无关；同时某一确定的 policy 是静态的，与时间无关，但是个体可以随着时间更新策略。

结合策略后，马尔科夫奖励过程转变为:

$$ P_{ss'}^{\pi} = \sum_{a \in \mathcal{A}} \pi(a|s) P_{ss'}^a  \tag{11} $$

奖励函数为:

$$ R_s^{\pi} = \sum_{a \in \mathcal{A}} \pi(a|s) R_s^a \tag{12}$$

回到式 $(2),(4)$，在状态 $s$ 下遵循策略 $\pi$ 的价值函数为:

$$ V_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q_{\pi}(s,a) \tag{13}$$

行为价值函数为:

$$
Q_{\pi}(s,a) = R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_{\pi}(s')
\tag{14}
$$

式 $(14)$ 表明，处于某一状态下的行为价值分为两部分:
- 离开这个状态的价值
- 进入新状态的价值期望

结合式$(13),(14)$ 可得:

$$
V_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) (R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_{\pi}(s'))
\tag{15.a}
$$

$$
Q_{\pi}(s,a) = R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \sum_{a \in \mathcal{A}} \pi(a'|s') Q_{\pi}(s',a')
\tag{15.b}
$$

结合策略，式 $(7)$ 转变为:

$$
\begin{equation}\begin{split} 
\boldsymbol{V}_{\pi} &= \boldsymbol{R}^{\pi} + \gamma \boldsymbol{P}^{\pi} \boldsymbol{V}_{\pi}\\
\boldsymbol{V}_{\pi} &=(\boldsymbol{I} - \gamma \boldsymbol{P}^{\pi})^{-1} \boldsymbol{R}^{\pi}\\ 
\end{split}\end{equation}
\tag{16}
$$
<!-- 展开后可得

$$V_{\pi}(s) =\sum_a \pi(a|s) \sum_{s'} \sum_r p(s',r|s,a) [r+ \gamma V_{\pi}(s')]$$ -->

<!-- 其中
- $\pi(a|s) = P(a|s)$ 表示从状态 $s$ 采取动作 $a$ 的概率， 也称 **策略**
- $P(s'|a, s)$ 表示在状态 $s$ 下采取动作 $a$ 后转移到状态 $s'$ 的概率
- $P(s'|s)$ 表示从状态 $s$ 下转移到状态 $s'$ 的概率  -->

## 参考
[知乎用户-量子位](https://www.zhihu.com/question/41775291/answer/602826860)
[知乎用户-张楚珩](https://zhuanlan.zhihu.com/p/56045177)
[知乎用户-刹那 Kevin](https://zhuanlan.zhihu.com/p/381821556)
[知乎用户-叶强](https://zhuanlan.zhihu.com/p/28084904)