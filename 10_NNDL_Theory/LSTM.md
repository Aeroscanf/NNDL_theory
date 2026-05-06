#深度学习笔记：长短期记忆网络 (LSTM)

> [!abstract] 笔记摘要
> 整理自邱锡鹏《神经网络与深度学习》第 6.6.1 节。
> LSTM 通过引入**门控机制 (Gating Mechanism)** 和**内部记忆单元 (Internal Memory Cell)**，有效解决了简单 RNN 中的梯度消失和长期依赖问题。

---

## 1. 核心数学公式

LSTM 的核心在于通过三个“门”（遗忘门、输入门、输出门）来控制信息在内部状态 $c_t$ 中的累积与遗忘。

### 1.1 门控计算 (Gates)
假设当前输入为 $x_t$，上一时刻的外部状态为 $h_{t-1}$，$\sigma$ 为 Sigmoid 激活函数：

1.  **遗忘门 (Forget Gate)**: 控制上一时刻的内部记忆 $c_{t-1}$ 有多少需要保留。
    $$ f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) $$
2.  **输入门 (Input Gate)**: 控制当前时刻的候选记忆 $\tilde{c}_t$ 有多少需要保存。
    $$ i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) $$
3.  **输出门 (Output Gate)**: 控制当前的内部记忆 $c_t$ 有多少需要输出给外部状态 $h_t$。
    $$ o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) $$

### 1.2 状态更新 (Update)
利用上述门控结果更新记忆单元和隐状态：

4.  **候选状态 (Candidate State)**: 当前时刻的新信息。
    $$ \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) $$
5.  **内部记忆更新 (Cell State)**: **(核心步骤)** 旧记忆的遗忘 + 新记忆的写入。
    $$ c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t $$
6.  **外部状态更新 (Hidden State)**: 输出门控制下的最终输出。
    $$ h_t = o_t \odot \tanh(c_t) $$

> [!info] 符号说明
> *   $W_*, U_*, b_*$: 可学习的权重矩阵和偏置。
> *   $\sigma(\cdot)$: Logistic Sigmoid 函数，输出区间 $(0, 1)$。
> *   $\odot$: 向量元素逐点相乘 (Element-wise product)。

---

## 2. PyTorch 风格伪代码

为了方便理解，以下代码将公式拆解为逐行实现。在实际生产环境（如 `torch.nn.LSTM`）中，通常会将矩阵运算合并以提高计算效率。

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        # 定义各个门的线性变换层 (Wx + b)
        # 实际实现中，通常会将这4个操作合并为一个大的矩阵乘法以加速
        self.xf = nn.Linear(input_size, hidden_size) # 遗忘门输入权重
        self.xi = nn.Linear(input_size, hidden_size) # 输入门输入权重
        self.xo = nn.Linear(input_size, hidden_size) # 输出门输入权重
        self.xc = nn.Linear(input_size, hidden_size) # 候选态输入权重

        # 定义隐状态的线性变换层 (Uh + b)
        self.hf = nn.Linear(hidden_size, hidden_size)
        self.hi = nn.Linear(hidden_size, hidden_size)
        self.ho = nn.Linear(hidden_size, hidden_size)
        self.hc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, states):
        """
        x_t: 当前时刻输入 [batch_size, input_size]
        states: 元组 (h_prev, c_prev)，上一时刻的隐状态和记忆单元
        """
        h_prev, c_prev = states

        # 1. 计算三个门 (Gates)
        # f_t: 遗忘门
        f_t = torch.sigmoid(self.xf(x_t) + self.hf(h_prev))
        # i_t: 输入门
        i_t = torch.sigmoid(self.xi(x_t) + self.hi(h_prev))
        # o_t: 输出门
        o_t = torch.sigmoid(self.xo(x_t) + self.ho(h_prev))

        # 2. 计算候选状态 (Candidate Cell)
        # c_tilde_t: 使用 tanh 激活
        c_tilde = torch.tanh(self.xc(x_t) + self.hc(h_prev))

        # 3. 更新记忆单元 (Cell State) -> 核心公式
        # c_t = (遗忘旧的) + (写入新的)
        c_t = f_t * c_prev + i_t * c_tilde

        # 4. 更新隐状态 (Hidden State)
        # h_t = 输出门 * tanh(当前记忆)
        h_t = o_t * torch.tanh(c_t)

        return h_t, (h_t, c_t)
````

## 3. 关键点总结

- **长短期记忆**：$c_t$ 就像一个计数器，通过线性的循环连接（$f_t \odot c_{t-1}$）可以保持长距离的信息而不发生梯度消失。
- **Sigmoid vs Tanh**：
    - **门 (Gate)** 使用 `Sigmoid` (0~1) 是为了作为“开关”或“比例”来控制信息流量。
    - **候选态/输出** 使用 `Tanh` (-1~1) 是为了生成正负值，允许状态增加或减少信息，且保持数值稳定性。
## 4.考点
- **考点一：遗忘门与梯度消失**

> 	 **问题**：为什么 LSTM 能避免梯度消失？
> 	**答案**：很大程度上归功于遗忘门。在简单 RNN 中，梯度反向传播涉及连乘 Wk，导致指数衰减。而在 LSTM 中，如果遗忘门 ft​≈1，记忆单元的梯度传播路径变为ct​=1×ct−1​+…，梯度可以**线性地、无损地**传回很久以前的时刻。这类似于` ResNet` 中的残差连接。

- **考点二：偏置初始化 (Bias Initialization) (高阶考点)**

	>• **问题**：训练 LSTM 时，遗忘门的偏置 bf​ 应该如何初始化？
	>• **答案**：**应该设得比较大（如 1 或 2）**，而不是像其他权重一样初始化为 0 或很小的值。
	>• **原因**：
		    ◦ 如果初始化较小，Sigmoid 输出的 ft​ 会接近 0.5 或更小。
		    ◦ 这意味着训练刚开始，网络就会倾向于“遗忘”大部分历史信息。
		    ◦ 这会导致梯度在反向传播时迅速衰减（梯度消失），网络很难捕捉长距离依赖。
		    ◦ **设为大值**是为了让训练初期 ft​≈1（门默认打开），先保住记忆，让网络在训练中慢慢学会“什么该忘”。

---

> **参考文献**
> 
> - 邱锡鹏. _神经网络与深度学习_. 第 6.6.1 节 长短期记忆网络.