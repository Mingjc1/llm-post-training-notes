# RLHF 入门：从语言模型到策略优化

> 根据我的 RLHF 学习材料整理。范围是经典的“偏好数据 → 奖励模型 → PPO”路线；DPO、GRPO 留待后续笔记。

## 1. 先把语言生成看成强化学习

给定提示词 $x$，模型逐个生成 token，最终得到回答 $y=(a_1,\ldots,a_T)$。在第 $t$ 步：

| 强化学习概念 | 语言模型中的对应 |
| --- | --- |
| 状态 $s_t$ | 提示词以及此前生成的 token，即 $(x,a_{<t})$ |
| 动作 $a_t$ | 下一枚 token（也可能是结束符） |
| 策略 $\pi_\theta$ | 模型给定当前上下文时对下一枚 token 的概率分布 |
| 轨迹 $y$ | 从第一枚到最后一枚 token 的完整回答 |
| 奖励 | 对回答的反馈；经典 RLHF 中通常包含完整回答的奖励模型分数，以及逐 token 的 KL 惩罚 |

因此，**LLM 本身就是策略**：$\pi_\theta(a_t\mid x,a_{<t})$。注意 token 是分词器的单位，未必是一个汉字或一个词；“每生成一个汉字就是一个动作”并不准确。

## 2. 为什么在 SFT 后引入偏好数据

SFT 用示范回答 $(x,y)$ 训练模型，提高该回答的条件概率：

$$
\mathcal L_{\mathrm{SFT}}=-\log \pi_\theta(y\mid x)
$$

开放式问题常有多种合格答案。偏好标注改为比较同一提示词下的两个回答：$y_w\succ y_l$，分别表示更受偏好的回答和较差的回答。它提供了“哪一个更好”的信号，但偏好仍受标注标准和数据质量影响，并不等同于客观真理。

## 3. 经典 RLHF 的三步

1. **SFT**：用示范回答得到初始策略 $\pi_{\mathrm{SFT}}$。
2. **训练奖励模型（RM）**：收集同一提示词下的回答偏好，让 $R_\phi(x,y)$ 为完整回答输出一个标量分数。
3. **PPO 优化策略**：从 SFT 模型出发，生成回答、取得奖励、估计优势，再更新生成模型。

### 奖励模型如何学习偏好

对一条偏好样本 $(x,y_w,y_l)$，常见的成对排序损失是：

$$
\mathcal L_{\mathrm{RM}}
=-\log\sigma\!\left(R_\phi(x,y_w)-R_\phi(x,y_l)\right)
$$

它鼓励较优回答的分数高于较差回答。**看分数差，不要把单个分数解释为绝对质量**：同一奖励模型的所有分数一起加上常数，损失不变；分数的正负也没有天然的“好/坏”含义。

### PPO 阶段的四个角色

| 角色 | 训练时做什么 |
| --- | --- |
| Actor / Policy $\pi_\theta$ | 生成回答；这是最终要更新的语言模型 |
| Reward Model $R_\phi$ | 给完整回答打分，提供学习信号 |
| Critic / Value $V_\psi(s_t)$ | 估计从当前生成位置出发的预期回报，辅助计算 advantage |
| Reference $\pi_{\mathrm{ref}}$ | 通常是冻结的 SFT 策略，提供与当前策略比较的基准 |

一次训练迭代可以记为：**采样 prompt → Actor 生成回答 → RM 给分、Reference 提供对照 → Critic 估计价值 → 计算 advantage → PPO 更新 Actor（并训练 Critic）**。RM 和 Critic 是不同的角色：前者评价完成的回答，后者估计生成过程中某个状态的预期回报。Critic 的训练目标通常也会包含 KL 惩罚后的回报。

## 4. KL 惩罚与 PPO 裁剪分别管什么

一个便于理解的策略目标是：

$$
\max_\theta\;\mathbb E_{x,\,y\sim\pi_\theta(\cdot\mid x)}
\left[R_\phi(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right]
$$

其中，$\log\pi(y\mid x)$ 可以拆成各生成 token 的对数概率之和。对当前策略生成的回答取期望时，对数比值对应策略与参考策略的 KL 散度。实现中常把该惩罚分配到各 token，RM 分数则通常在回答结束时给出。

**参考模型的 KL 惩罚**约束策略长期偏离 SFT 模型，减轻只迎合奖励模型而造成的异常行为。**PPO 的裁剪**比较当前策略与*采样这些回答时的旧策略*，限制一次更新的幅度。两者的比较对象不同，不能混为一个约束。奖励模型可能被“钻空子”（reward hacking），因此即使 RM 分数上升，也应检查回答质量。

## 5. 与已学强化学习概念对上

- **Policy Gradient**：增加高优势动作的概率，降低低优势动作的概率；这里的动作是生成的 token。
- **Actor–Critic**：Actor 负责生成，Critic 的价值估计用于计算优势、降低梯度估计的方差。
- **Advantage**：表示某个动作相对当前状态下预期水平有多好。实践中可用 GAE 等方法估计，不需要为每个 token 单独训练一个奖励模型。
- **PPO**：用基于旧策略与新策略概率比的裁剪目标更新 Actor；KL 惩罚是 RLHF 中另外加入的参考约束。

## 6. 复习时自问

1. 在生成第 $t$ 枚 token 时，状态、动作和策略分别是什么？
2. SFT 的示范数据与 RM 的成对偏好数据分别提供什么信号？
3. 为什么奖励模型分数不能直接解释成“5 分比 1 分绝对好 4 分”？
4. RM、Critic、Reference 各自输出什么，分别用于哪一步？
5. PPO 裁剪与参考模型 KL 惩罚分别比较哪两个策略？

## 资料

- 我的原始 RLHF 学习材料（本笔记据此整理，并修正了几处简化表述）。
- [Ouyang et al., *Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155)：SFT、偏好排序 RM、PPO 与参考策略 KL 惩罚。
- [Schulman et al., *Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)：PPO 的裁剪目标。
