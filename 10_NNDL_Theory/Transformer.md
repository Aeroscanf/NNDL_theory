

# 深度学习笔记：Transformer 模型架构详解

> [!abstract] 笔记摘要
> 本笔记整理自邱锡鹏《神经网络与深度学习》第 15.6.3 节。
> **核心主题**：Transformer 是一个基于**自注意力机制（Self-Attention）**的序列到序列（Seq2Seq）模型。它摒弃了传统的循环神经网络（RNN）和卷积神经网络（CNN）结构，通过全连接的自注意力机制来捕捉长距离依赖关系，并实现了并行计算。

---

## 1. 架构图解说 (基于图 15.6)

Transformer 的整体架构由 **编码器 (Encoder)** 和 **解码器 (Decoder)** 两部分组成。
![[Pasted image 20260122203405.png]]
### 1.1 左侧：编码器 (Encoder)
编码器的作用是将输入序列转化为连续的特征向量表示。

*   **输入层 (Input Embedding + Positional Encoding)**
    *   将输入词序列映射为向量。
    *   **位置编码**：由于自注意力机制忽略了序列的位置信息（即交换输入顺序不改变对内容的理解），必须在输入端显式地将位置信息（Positional Encoding）加到词嵌入向量上，让模型感知词序。
*   **堆叠层 ($N\times$)**
    *   由 $N$ 个相同的层堆叠而成，每一层包含两个子层：
        1.  **多头自注意力 (Multi-Head Attention)**：在不同的特征子空间中并行捕捉词与词之间的依赖关系。
        2.  **前馈神经网络 (Feed Forward)**：对每个位置的向量进行非线性变换。
*   **连接机制 (Add & Norm)**
    *   每个子层周围都采用了**残差连接 (Residual Connection)** 和 **层归一化 (Layer Normalization)**。
    *   公式：$\text{LayerNorm}(x + \text{Sublayer}(x))$。

### 1.2 右侧：解码器 (Decoder)
解码器的作用是根据编码器的输出，自回归地生成目标序列。

*   **输入层 (Output Embedding + Positional Encoding)**
    *   输入是“右移一位”的目标序列（Outputs shifted right），同样需要加上位置编码。
*   **堆叠层 ($N\times$)**
    *   每一层包含三个子层：
        1.  **掩蔽多头注意力 (Masked Multi-Head Attention)**：这是解码器的自注意力层。**Masked (掩蔽)** 是关键，它通过掩码机制屏蔽掉当前位置之后的“未来”信息，确保模型在预测第 $t$ 个词时只能看到 $t$ 之前的词，保证自回归生成的因果性。
        2.  **交互注意力 (Multi-Head Attention)**：也称为编码器-解码器注意力。这里的 **Query** 来自解码器上一层的输出，而 **Key** 和 **Value** 来自编码器的最终输出。这一层让解码器在生成每个词时，都能“查询”输入序列中最相关的信息。
        3.  **前馈神经网络 (Feed Forward)**：与编码器类似。
*   **输出层**
    *   通过 **Linear** 层映射维度，再经过 **Softmax** 层计算输出词表中每个词的概率。

---

## 2. Self-Attention 核心公式

根据书中公式 (15.107) - (15.111) 整理。

### 2.1 线性投影 (Linear Projections)
假设输入序列为 $H$，首先通过三个可学习的投影矩阵将其映射到查询 (Query)、键 (Key)、值 (Value) 空间：

$$
\begin{aligned}
Q &= W_Q H \\
K &= W_K H \\
V &= W_V H
\end{aligned}
$$

> 注：$H$ 的每一列代表一个时间步的输入向量。$W_Q, W_K, W_V$ 为可学习的参数矩阵。

### 2.2 缩放点积注意力 (Scaled Dot-Product Attention)
这是 Transformer 的核心计算单元。计算查询与键的相似度，归一化后对值进行加权求和：

$$
\text{self-att}(Q, K, V) = \text{softmax}\left( \frac{K^T Q}{\sqrt{d_k}} \right) V
$$

*   **$K^T Q$**：计算注意力得分（相似度）。
*   **$\frac{1}{\sqrt{d_k}}$**：**缩放因子**。其中 $d_k$ 是键向量的维度。除以 $\sqrt{d_k}$ 是为了防止点积数值过大导致 Softmax 进入饱和区（梯度消失），从而稳定训练。

### 2.3 多头注意力 (Multi-Head Attention)
为了增强模型的表达能力，Transformer 并行计算 $M$ 个注意力头，每个头关注不同的信息子空间：

1.  **分头计算**：对于第 $m$ 个头，使用独立的参数矩阵计算：
    $$ \text{head}_m = \text{self-att}(Q_m, K_m, V_m) $$
2.  **拼接与融合**：将所有头的输出拼接起来，通过输出矩阵 $W_O$ 进行线性变换：
    $$ \text{MultiHead}(H) = W_O [\text{head}_1; \dots; \text{head}_M] $$

---

> **参考来源**
> *   邱锡鹏. *神经网络与深度学习*. 机械工业出版社. 第 15 章 序列生成模型 (15.6.3 基于自注意力的序列到序列模型).
