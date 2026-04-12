# 第三章《大语言模型基础》笔记

## 一、章节总览

本章核心问题是：**现代智能体为什么能工作，大语言模型为什么会表现出强大的理解、生成与一定程度的推理能力？**

全章主线可以概括为：

1. 语言模型的基本任务是什么
2. 语言模型如何从统计方法演进到神经网络，再到 Transformer
3. 为什么今天的大模型大多采用 **Decoder-only** 架构
4. 我们如何通过 **Prompt、采样参数、Tokenizer** 与模型有效交互
5. 如何调用和部署开源模型
6. 如何做模型选型
7. 大模型为什么会越来越强（缩放法则），又为什么仍然会出错（幻觉等局限）

---

# 二、语言模型基础：从“统计计数”到“深度建模”

## 2.1 语言模型的本质

语言模型（Language Model, LM）的核心任务是：

> **为一个词序列出现的概率建模，或者等价地，预测下一个最可能出现的词。**

对于句子 $S = w_1,w_2,\dots,w_m$，其概率可写为：

$$
P(S)=P(w_1)\cdot P(w_2|w_1)\cdot P(w_3|w_1,w_2)\cdots P(w_m|w_1,\dots,w_{m-1})
$$

这就是**链式法则**。

### 我的理解
语言模型本质上不是“理解世界”，而是在做一个极其强大的概率预测器：
- 输入已有上下文
- 输出下一个 token 的概率分布
- 连续重复这个过程，就能生成文本

---

## 2.2 N-gram：最早期的语言模型思路

### 2.2.1 核心思想

直接估计完整条件概率太难，所以引入**马尔可夫假设**：

> 一个词的出现概率，只依赖它前面有限个词，而不是全部历史。

于是得到 **N-gram 模型**：
- **Bigram**：只看前 1 个词
- **Trigram**：只看前 2 个词
- 更一般地：只看前 $n-1$ 个词

例如：

$$
P(w_i|w_1,\dots,w_{i-1}) \approx P(w_i|w_{i-1})
$$

或

$$
P(w_i|w_1,\dots,w_{i-1}) \approx P(w_i|w_{i-2},w_{i-1})
$$

### 2.2.2 参数估计方法

使用**最大似然估计（MLE）**，本质就是计数：

对于 Bigram：

$$
P(w_i|w_{i-1})=\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
$$

即：
- 分子：词对连续出现的次数
- 分母：前一个词出现的总次数

### 2.2.3 优点与局限

#### 优点
- 简单直观
- 易于计算
- 在小规模任务中有一定效果

#### 致命问题
1. **数据稀疏性**
   - 没见过的词序列概率直接变成 0
   - 需要平滑，但只能缓解，不能根治

2. **泛化能力差**
   - 模型把词当作离散符号
   - 无法理解 `agent` 和 `robot` 语义相近
   - 见过 `agent learns`，也无法自然迁移到 `robot learns`

### 结论
N-gram 的问题，决定了它不适合现代大模型，但它非常重要，因为它明确了语言模型的原始目标：

> **预测序列概率 / 预测下一个词**

---

## 2.3 神经网络语言模型：引入词向量

N-gram 最大的问题，是把词看成孤立的符号。  
神经网络语言模型的关键突破在于：

> **用连续向量表示词，而不是用离散 ID。**

### 2.3.1 词嵌入（Word Embedding）

每个词被映射到一个高维连续空间中的向量：
- 语义接近的词，向量接近
- 语义差异大的词，向量距离远

例如：
- `agent` 和 `robot` 更接近
- `agent` 和 `apple` 更远

### 2.3.2 余弦相似度

衡量两个词向量是否接近，常用余弦相似度：

$$
\text{similarity}(\vec{a},\vec{b})=\frac{\vec{a}\cdot \vec{b}}{|\vec{a}||\vec{b}|}
$$

- 越接近 1：越相似
- 接近 0：关系弱
- 接近 -1：方向相反

### 2.3.3 词向量的意义

词嵌入不只是“压缩表示”，更重要的是它让模型第一次具备了：

- 语义泛化能力
- 近义词迁移能力
- 一定的关系建模能力

经典例子：

`king - man + woman ≈ queen`

### 2.3.4 局限

虽然神经网络语言模型解决了 N-gram 的泛化问题，但它仍然有一个明显限制：

> **上下文窗口是固定的。**

它只能看固定长度的前文，长距离依赖处理不好。

---

# 三、RNN / LSTM：让模型拥有“记忆”

## 3.1 RNN 的核心思想

循环神经网络（RNN）引入了**隐藏状态（hidden state）**，可以理解为模型的“短期记忆”。

在每一个时间步：
- 输入当前 token
- 结合上一时刻隐藏状态
- 产生新的隐藏状态
- 将新状态传给下一个时间步

因此它第一次能够处理：

> **可变长度序列**

### 直观理解
N-gram / 前馈网络：固定窗口看几步  
RNN：理论上可以一直把历史信息往后传

---

## 3.2 RNN 的问题：梯度消失与长距离依赖

虽然 RNN 有“记忆”，但这个记忆并不稳定。

随着序列变长，早期信息在多次传递中会逐渐衰减，出现：
- 梯度消失
- 长距离依赖难以学习
- 训练效率低
- 无法并行

这意味着：

> RNN 理论上能记很久，实际上很难真的记住很久。

---

## 3.3 LSTM：改进版记忆机制

为了解决 RNN 的长距离依赖问题，提出了 **LSTM（Long Short-Term Memory）**。

### 核心思想
LSTM 通过门控机制控制信息流动：
- 忘记什么
- 保留什么
- 输出什么

常见理解为三个门：
1. 忘记门
2. 输入门
3. 输出门

### 作用
LSTM 可以更稳定地保留长期信息，因此相比普通 RNN 更适合处理长序列。

---

## 3.4 为什么今天不再以 RNN/LSTM 为主流

RNN/LSTM 在历史上非常重要，但今天在大语言模型中已不是主角，原因主要有两点：

1. **串行计算，难并行**
   - 第 t 步必须等第 t-1 步
   - 不适合大规模训练

2. **长距离依赖依然不够理想**
   - 即使用 LSTM，也不如后来的注意力机制直接高效

### 结论
RNN/LSTM 是通往 Transformer 的关键过渡，但在当代 LLM 中，理解其思想即可，工程上重点应转向 Transformer。

---

# 四、Transformer：现代大语言模型的基础

## 4.1 Transformer 为什么重要

Transformer 是现代 LLM 的基础架构。  
它最大的突破是：

> **放弃循环结构，用注意力机制直接建模任意位置之间的关系。**

这带来两大优势：
1. 更容易捕捉长距离依赖
2. 可以并行训练

---

## 4.2 自注意力（Self-Attention）

### 4.2.1 直觉

处理一个词时，模型不只看它自己，而是会关注句子中其他相关词。

例如：
> “The agent learns because it is intelligent.”

理解 `it` 时，模型需要更多关注 `agent`。

### 4.2.2 Q、K、V 机制

每个 token 会映射成三个向量：
- **Q（Query）**：我要找什么信息
- **K（Key）**：我有哪些标签可供匹配
- **V（Value）**：我真正携带的信息内容

### 4.2.3 计算流程

1. 对每个 token 生成 Q、K、V
2. 用当前 token 的 Q 去和所有 token 的 K 做点积
3. 得到相关性分数
4. 除以 $\sqrt{d_k}$ 进行缩放
5. 经过 Softmax 归一化
6. 用这些权重对所有 V 做加权求和
7. 得到当前 token 的新表示

公式：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 4.2.4 本质理解

自注意力做的事可以概括为：

> **让每个 token 动态决定：在理解自己时，应该参考上下文中的哪些 token、参考多少。**

---

## 4.3 多头注意力（Multi-Head Attention）

单个注意力头只能从一种角度看关系。  
多头注意力的思想是：

> **并行使用多个注意力头，让模型从不同子空间、不同角度理解同一序列。**

可能有的头关注：
- 语法依赖
- 指代关系
- 语义相似
- 局部搭配
- 长距离关联

### 作用
多头注意力提升了表达能力，使模型可以同时捕捉多种关系模式。

---

## 4.4 前馈网络（Feed Forward Network, FFN）

在每一层注意力后，Transformer 还会接一个位置前馈网络。

它的作用不是建模 token 之间关系，而是：

> **对每个 token 的表示再做一次非线性变换和特征提炼。**

可以理解为：
- 注意力负责“信息聚合”
- FFN 负责“特征加工”

---

## 4.5 残差连接与层归一化

Transformer 训练深层网络时，还依赖两个关键结构：

### 4.5.1 残差连接（Residual Connection）
公式可写成：

$$
\text{Output} = x + \text{Sublayer}(x)
$$

作用：
- 缓解梯度消失
- 保证深层网络更容易训练
- 保留原始信息通路

### 4.5.2 层归一化（LayerNorm）
作用：
- 稳定每层输入分布
- 加快收敛
- 提高训练稳定性

---

## 4.6 位置编码（Positional Encoding）

注意力本身只看 token 间关系，不感知顺序。

因此：
- `agent learns`
- `learns agent`

对纯注意力而言可能被看成相似结构。

所以 Transformer 需要显式引入**位置信息**。

### 原始做法
使用固定的正弦 / 余弦位置编码：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

### 作用
让模型知道：
- token 的绝对位置
- 一定程度上的相对位置信息

### 现实补充
原章讲的是经典正弦位置编码。  
但在今天的实际大模型中，**位置机制仍然是关键问题**，因为它直接影响：
- 长上下文能力
- 外推长度能力
- 推理与生成稳定性

---

# 五、Encoder-Decoder 与 Decoder-Only

## 5.1 完整 Transformer 的基本结构

经典 Transformer 一般分为：
- **Encoder**
- **Decoder**

### Encoder 的作用
对输入序列做深度理解，形成上下文表示。

### Decoder 的作用
基于上下文表示，自回归地生成输出序列。

这种架构很适合：
- 机器翻译
- 摘要
- 结构化输入到文本输出

---

## 5.2 Decoder-Only 架构

今天的大语言模型多数采用 **Decoder-only** 架构。

核心思想：

> **语言生成的本质就是根据已有上下文，预测下一个 token。**

因此只保留解码器即可。

### 工作方式：自回归生成
过程如下：
1. 输入已有文本
2. 预测下一个 token
3. 将预测结果拼回上下文
4. 再预测下一个 token
5. 不断重复直到结束

这就是所谓的 **Autoregressive（自回归）**。

---

## 5.3 为什么 Decoder-Only 成为主流

这是本章最值得重点掌握的现代结论之一。

### 原因 1：目标统一
一个“预测下一个 token”的统一目标，可以覆盖很多任务：
- 对话
- 写作
- 摘要
- 问答
- 代码生成
- 工具调用
- Agent 规划输出

### 原因 2：训练简单且可扩展
相比复杂的 seq2seq 设计，Decoder-only 更统一，适合海量数据预训练。

### 原因 3：更符合通用生成范式
今天我们使用大模型时，大多数任务都可以转化为：

> 给定上下文，让模型继续往后写。

---

## 5.4 掩码机制（Masking）

Decoder-only 里有一个关键点：  
模型虽然在训练时能看到整个序列，但在预测某个位置时，**不能偷看未来 token**。

因此需要使用**因果掩码（causal mask）**：
- 屏蔽当前位置之后的 token
- 保证每个位置只能关注自己左边的内容

### 本质
这保证了训练目标和实际生成过程保持一致。

---

## 5.5 现阶段最值得记住的架构结论

### 历史脉络
- N-gram：统计计数
- 神经网络 LM：引入词向量
- RNN/LSTM：引入时序记忆
- Transformer：引入注意力与并行训练
- Decoder-only：成为现代通用生成模型主流



为了真正理解 Transformer 的工作原理，最好的方法莫过于亲手实现它。

```Python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    为输入序列的词嵌入向量添加位置编码。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 偶数维度使用 sin, 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self,Q,K,V,mask=None):
        # 1. 计算注意力得分 (Q x K^T)
        attn_scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)

        # 2. 应用掩码 (如果提供)
        if mask is not None:
            # 将掩码中为 0 的位置设置为一个非常小的负数，这样 softmax 后会接近 0
            attn_scores = attn_scores.masked_fill(mask==0,-1e9)

        # 3. 计算注意力权重 (Softmax)
        attn_probs = torch.softmax(attn_scores,dim=-1)

        # 4. 加权求和 (权重 * V)
        output = torch.matmul(attn_probs,V)
        return output

    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size,seq_length,d_model = x.size()
        return x.view(batch_size,seq_length,self.num_heads,self.d_k).transpose(1,2)

    def combine_heads(self, x):
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size,num_heads,seq_length,d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. 对 Q, K, V 进行线性变换
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 2. 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q,K,V,mask)

        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output



class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionWiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x

# --- 编码器核心层 ---

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention() #待实现
        self.feed_forward = PositionWiseFeedForward() 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
        # 1. 多头自注意力
        attn_output = self.self_attn(x,x,x,mask)
        x = self.norm1(x+self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x+self.dropout(ff_output))

        return x

# --- 解码器核心层 ---

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        # 1. 掩码多头自注意力 (对自己)
        attn_output = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 交叉注意力 (对编码器输出)
        corss_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(corss_attn_output))

        #3.前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

```
### 工程上最重要的一句话
> **今天做通用大模型 / 智能体，优先理解 Decoder-only Transformer，而不是把精力放在 RNN。**

---

# 六、提示工程（Prompt Engineering）

## 6.1 Prompt 的本质

如果把模型看作一个非常强的概率生成器，那么 Prompt 就是：

> **给这个生成器设定任务边界、角色、格式和上下文的方式。**

对于智能体来说，Prompt 直接影响：
- 任务理解
- 输出风格
- 工具调用意图
- 协作模式
- 稳定性

---

## 6.2 采样参数：控制输出风格与稳定性

这是今天非常实用的一部分。

模型并不是总选概率最大的 token，它可以通过采样参数改变输出风格。

---

### 6.2.1 Temperature

Softmax 概率分布会受到温度参数影响。

#### 作用
- **低温度**：分布更尖锐，输出更保守、更稳定
- **高温度**：分布更平滑，输出更多样、更发散

#### 使用建议
- 事实问答 / 信息抽取 / 代码修复：低温
- 创意写作 / 头脑风暴 / 文案生成：高温

#### 经验理解
- `temperature = 0`：几乎最确定
- `0.2 ~ 0.5`：偏稳定
- `0.7 ~ 1.0`：平衡创造性与可控性
- 更高：更发散，但也更容易跑偏

---

### 6.2.2 Top-k

先按概率排序，只保留前 k 个 token，再重新归一化采样。

#### 特点
- 直接限制候选集规模
- `k=1` 时退化为贪心解码

#### 适合
- 需要控制输出随机性的场景

---

### 6.2.3 Top-p（核采样）

不固定保留多少个 token，而是：
- 从高概率 token 开始累计
- 直到概率和达到阈值 p
- 用这个最小集合做采样

#### 特点
- 相比 Top-k，更能动态适应概率分布
- 对“长尾”分布更灵活

---

### 6.2.4 参数选择经验

通常：
- `Top-k` 与 `Top-p` 二选一即可
- 如果强调稳定性，优先低温
- 如果强调自然多样性，常用 `Top-p`
- `temperature=0` 时，随机性几乎消失，Top-k / Top-p 的意义也会显著降低

### 实践结论
采样参数不是“调味品”，而是直接影响系统行为的控制杆。

---

## 6.3 Zero-shot / One-shot / Few-shot

这是提示设计的经典三种方式。

### 6.3.1 Zero-shot
不给示例，直接下指令。

#### 优点
- 简洁
- 成本低
- 通用

#### 缺点
- 对复杂格式任务不够稳

---

### 6.3.2 One-shot
给一个示例，让模型模仿格式。

#### 作用
- 帮模型理解输出样式
- 减少格式错误

---

### 6.3.3 Few-shot
给多个示例，覆盖边界情况。

#### 优点
- 对分类、抽取、结构化输出更稳定
- 对复杂任务更容易建立“隐式规则”

#### 缺点
- 占上下文
- 增加 token 成本

---

## 6.4 提示工程的现实理解

在今天的智能体系统中，Prompt 往往不仅仅是“一句话需求”，而是一个组合结构：

1. 系统角色设定
2. 任务描述
3. 输出格式约束
4. 可用工具说明
5. 示例
6. 外部检索到的上下文
7. 历史对话

### 重点结论
Prompt 工程的核心并不是“写得华丽”，而是：

> **把任务边界、上下文、格式要求和成功标准写清楚。**

---

# 七、文本分词（Tokenization）

## 7.1 为什么必须分词

计算机不能直接处理自然语言文本，只能处理数字。  
因此文本必须先变成 token，再映射成 ID。

也就是说：

> **模型真正看到的不是“句子”，而是 token 序列。**

---

## 7.2 为什么不能直接按“字”或“词”处理

### 按字符切分的问题
- 序列太长
- 语义粒度太小
- 建模效率低

### 按单词切分的问题
- 词表会非常大
- 未登录词（OOV）问题严重
- 新词、罕见词难处理

---

## 7.3 子词分词（Subword Tokenization）

现代大模型通常采用**子词分词**。

核心思想：

> 它的核心思想是：将常见的词（如 "agent"）保留为完整的词元，同时将不常见的词（如 "Tokenization"）拆分成多个有意义的子词片段（如 "Token" 和 "ization"）。这样既控制了词表的大小，又能让模型通过组合子词来理解和生成新词。

例如：
- 常见词：直接保留
- 罕见长词：拆成多个子词

### 好处
1. 控制词表规模
2. 减少 OOV 问题
3. 兼顾表达能力和泛化能力
4. 可以通过组合处理新词

---

## 7.4 BPE（Byte Pair Encoding）

BPE 是最常见的子词算法之一，也是学习 tokenizer 时最值得掌握的算法。

### 基本过程
1. 初始词表设为字符级
2. 统计语料中相邻 token 对出现频率
3. 找到最高频的一对，合并成新 token
4. 重复，直到词表达到预设大小

### 本质
BPE 是一个“从字符逐步学习常见片段”的贪心合并过程。

---

## 7.5 WordPiece 与 SentencePiece

### WordPiece
- 与 BPE 类似
- 合并标准更偏向提升语言模型整体概率

### SentencePiece
- 把空格也作为普通符号
- 分词与解码更可逆
- 对多语言更友好

---

## 7.6 为什么 Tokenizer 对开发者很重要

这一点在实际开发里非常关键。

### 影响 1：成本
很多 API 按 token 计费，不是按“字数”或“句子数”计费。

### 影响 2：上下文长度
模型能处理的是 token 数，不是自然语言字符数。

### 影响 3：输出稳定性
格式化文本、代码、表格、JSON，在不同 tokenizer 下切分效果不同，会影响生成表现。

### 影响 4：多语言表现
中文、英文、混合文本、专业术语，在分词后的 token 密度可能差异很大。

### 一个非常实用的结论
> 做 LLM 应用时，要逐渐养成“按 token 思考”的习惯，而不是只按字符或字数思考。

---

# 八、调用与部署开源大语言模型

## 8.1 为什么不只用 API

通过 API 调模型当然方便，但并不是唯一方式。  
本地部署开源模型的价值在于：

1. **隐私与数据安全**
2. **离线使用**
3. **更强可控性**
4. **可能更低的长期成本**
5. **便于定制化与微调**

---

## 8.2 Hugging Face Transformers 的意义

Transformers 提供了标准化接口来：
- 加载模型
- 加载 tokenizer
- 进行推理
- 进行微调
- 统一不同模型的使用方式

它是开源模型生态的核心基础设施之一。

---

## 8.3 本章示例的意义

一个小参数量对话模型做本地运行示例，目的是让读者先理解完整调用链路，而不是追求极限性能。
在 `transformers` 库中，我们通常使用 `AutoModelForCausalLM` 和 `AutoTokenizer` 这两个类来自动加载与模型匹配的权重和分词器。下面这段代码会自动从 Hugging Face Hub 下载所需的模型文件和分词器配置，这可能需要一些时间，具体取决于你的网络速度。

```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成！")
```

我们来创建一个对话提示，Qwen1.5-Chat 模型遵循特定的对话模板。然后，可以使用上一步加载的 `tokenizer` 将文本提示转换为模型能够理解的数字 ID（即 Token ID）。

```Python
# 准备对话输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍你自己。"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 编码输入文本
model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("编码后的输入文本:")
print(model_inputs)

>>>
{'input_ids': tensor([[151644, 8948, 198, 2610, 525, 264,  10950, 17847, 13,151645, 198, 151644, 872, 198, 108386, 37945, 100157, 107828,1773, 151645, 198, 151644, 77091, 198]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')}
```

现在可以调用模型的 `generate()` 方法来生成回答了。模型会输出一系列 Token ID，这代表了它的回答。

最后，我们需要使用分词器的 `decode()` 方法，将这些数字 ID 翻译回人类可以阅读的文本。

```Python
# 使用模型生成回答
# max_new_tokens 控制了模型最多能生成多少个新的Token
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# 将生成的 Token ID 截取掉输入部分
# 这样我们只解码模型新生成的部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的 Token ID
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n模型的回答:")
print(response)

>>>
我叫通义千问，是由阿里云研发的预训练语言模型，可以回答问题、创作文字，还能表达观点、撰写代码。我主要的功能是在多个领域提
供帮助，包括但不限于:语言理解、文本生成、机器翻译、问答系统等。有什么我可以帮到你的吗？
```

从学习角度，本节重点不在于“某个具体模型版本”，而在于掌握下面这条通路：

1. 选择模型
2. 下载模型权重和 tokenizer
3. 构造输入
4. 调用生成接口
5. 调整采样参数
6. 观察输出差异

---

## 8.4 学习本地部署时应重点掌握什么

### 必须掌握
- `tokenizer` 的作用
- `model.generate()` 的基本逻辑
- `max_new_tokens`、`temperature`、`top_p` 等参数影响
- 显存 / 内存与模型大小的关系
- 推理速度与参数规模的关系

### 不要过度纠结
- 某个入门示例中的具体型号是否“最强”
- 某个版本是否绝对最新

### 更重要的是
> 学会一套可迁移的方法，而不是死记一个模型名。

---

# 九、模型选型：智能体开发中的关键决策

## 9.1 模型选择不是“越大越好”

模型选型是一个典型的多目标平衡问题，要在下面几个维度之间折中：

- 性能
- 成本
- 延迟
- 可部署性
- 可控性
- 隐私
- 上下文窗口
- 多模态能力
- 工具调用能力

---

## 9.2 关键考量维度

### 9.2.1 性能与能力
不同模型擅长方向不同：
- 推理
- 代码
- 对话
- 创意写作
- 多语言
- 长文本理解

### 9.2.2 成本
闭源模型：
- API 按 token 计费

开源模型：
- 需要本地硬件和运维成本

### 9.2.3 延迟
实时交互任务尤其关注：
- 首 token 延迟
- 总响应时间
- 并发吞吐能力

### 9.2.4 上下文窗口
长文档分析、长对话记忆、多文件推理都依赖上下文窗口。

### 9.2.5 隐私与合规
企业内部数据、医疗、金融等场景通常更重视：
- 数据不出域
- 审计能力
- 部署自主权

### 9.2.6 可控性
是否能：
- 本地部署
- 微调
- 蒸馏
- 加 guardrails
- 接入私有知识库

---

## 9.3 闭源模型 vs 开源模型

### 闭源模型的典型特点
#### 优点
- 通常性能强
- API 易用
- 生态成熟
- 多模态、工具调用、推理能力往往更完整

#### 缺点
- 成本可能高
- 自主控制弱
- 隐私与合规受限
- 某些场景下难以深度定制

---

### 开源模型的典型特点
#### 优点
- 可本地部署
- 可微调
- 可审查权重与行为
- 社区生态活跃
- 对企业私有化场景友好

#### 缺点
- 部署门槛更高
- 推理优化和工程能力要求更强
- 在某些最前沿能力上可能仍略弱于顶级闭源模型

---

## 9.4 当前更实用的选型思路

### 如果你是做学习 / 原型验证
优先：
- 小中型开源模型
- 云端 API 快速验证
- 重点验证 Prompt、流程、工具链

### 如果你是做企业 Agent
重点看：
- 成本可控
- 工具调用能力
- RAG 效果
- 幻觉率
- 延迟
- 合规与私有化

### 如果你是做复杂高价值任务
更看重：
- 推理能力
- 稳定性
- 长上下文
- 结构化输出能力
- 外部工具协同能力

---

# 十、缩放法则（Scaling Laws）

## 10.1 什么是缩放法则

缩放法则指出：

> 模型性能与参数量、数据量、计算量之间存在可预测的幂律关系。

简单说：
- 模型更大
- 数据更多
- 训练算力更强

通常模型性能就会更稳定地提升。

---

## 10.2 为什么缩放法则重要

它说明大模型的进步不是偶然的，而是具有系统性的。

这解释了为什么过去几年里模型能力持续跃迁：
- 参数量扩大
- 训练数据暴涨
- 算力持续投入
- 工程优化不断进步

---

## 10.3 Chinchilla 视角：不是只堆参数就行

缩放法则的进一步修正告诉我们：

> **参数量与数据量之间存在更优配比。**

也就是说：
- 不是模型越大越好
- 如果数据不足，盲目增大模型并不划算
- 数据效率同样重要

### 启示
现代大模型训练不只是“拼参数规模”，而是：
- 参数
- 数据
- 计算预算
三者一起优化

---

## 10.4 能力涌现（Emergence）

当模型规模足够大时，会出现小模型没有或不明显的能力，例如：
- 更强的指令遵循
- 多步推理
- 更好的代码生成
- 更复杂的任务迁移能力

### 我的理解
所谓“涌现”，可以理解为：
- 能力不是线性增加的
- 到达某个规模阈值后，系统会出现新的行为模式

---

# 十一、大语言模型的局限性

## 11.1 幻觉（Hallucination）

这是今天做 LLM 应用最必须警惕的问题之一。

### 定义
模型生成了：
- 与事实不符的内容
- 与输入矛盾的内容
- 不存在的实体、数据、事件
- 不忠实于原文的总结 / 翻译 / 改写

### 常见类型
1. **事实性幻觉**
   - 与现实世界事实不符

2. **忠实性幻觉**
   - 没有忠实反映输入内容

3. **内在幻觉**
   - 与给定上下文直接矛盾

---

## 11.2 幻觉为什么会发生

### 原因 1：训练数据并不完美
数据可能包含：
- 错误
- 矛盾
- 过时信息
- 偏见

### 原因 2：生成机制决定了它本质上在“预测 token”
模型没有天然的事实核查模块。

### 原因 3：复杂推理链容易出错
在长链条推理中，一步出错可能步步错。

---

## 11.3 其他局限

### 11.3.1 知识时效性
模型知识通常停留在训练数据截止时点之前。

### 11.3.2 偏见问题
模型可能吸收训练数据中的社会偏见与刻板印象。

### 11.3.3 不确定性表达不足
模型往往会“很自信地说错”。

---

## 11.4 缓解幻觉的方法

这是非常现代、非常实用的一节。

### 11.4.1 数据层面
- 数据清洗
- 引入更高质量事实知识
- RLHF 等对齐方法

### 11.4.2 模型层面
- 改进架构
- 引入不确定性表达机制

### 11.4.3 推理与生成层面（最实用）
#### 1）RAG（检索增强生成）
先检索外部知识，再把检索结果作为上下文喂给模型。

**适用场景：**
- 企业知识库问答
- 文档助手
- 论文阅读助手
- 客服系统

#### 2）多步推理与验证
让模型先分步分析，再检查过程或结论。

**适用场景：**
- 数学
- 复杂推理
- 多条件决策
- 流程性任务

#### 3）外部工具调用
让模型调用：
- 搜索引擎
- 计算器
- 数据库
- 代码执行器
- API

**适用场景：**
- 实时信息查询
- 精确计算
- 数据处理
- Agent 执行动作

### 关键结论
> 想降低幻觉，不能只“相信模型更聪明”，而要让系统具备检索、验证和工具使用能力。

---

# 十二、这一章最值得记住的现代结论

## 12.1 架构层面
现代大语言模型的核心基础是：

> **Transformer，尤其是 Decoder-only 的自回归生成范式。**

---

## 12.2 使用层面
与模型交互时，最重要的是：

1. Prompt 是否清晰
2. 采样参数是否合理
3. Tokenizer 与上下文成本是否被正确理解

---

## 12.3 系统层面
做真正可用的智能体，不能只靠“裸模型”，而要组合：

- Prompt
- 检索
- 工具
- 记忆
- 工作流
- 评测

---

## 12.4 工程层面
模型选型不是选“最强名词”，而是选：
- 任务匹配度
- 成本可承受
- 延迟可接受
- 幻觉可控制
- 部署可实现

---

# 十三、我自己的总结

这一章最核心的价值，不是让人死记公式，而是建立一个完整认知框架：

## 13.1 认知框架一：语言模型的本质
大模型不是神秘黑箱，本质上是一个强大的“下一个 token 预测器”。

## 13.2 认知框架二：能力来自哪里
能力来自：
- 大规模数据
- 大规模参数
- 大规模计算
- 合适架构
- 有效对齐与推理流程

## 13.3 认知框架三：为什么今天是 Transformer / Decoder-only
因为它更适合：
- 并行训练
- 长距离依赖建模
- 统一生成任务
- 大规模预训练扩展

## 13.4 认知框架四：为什么单有模型还不够
因为模型会：
- 幻觉
- 过时
- 受上下文限制
- 受采样策略影响

所以现代 Agent 系统必须依赖：
- RAG
- 工具调用
- 结构化提示
- 验证与反馈机制

---

# 十四、复习时建议重点背的内容

## 必背 1：语言模型本质
- 序列概率建模
- 下一个 token 预测

## 必背 2：模型演进链条
- N-gram
- 神经网络语言模型
- RNN/LSTM
- Transformer
- Decoder-only

## 必背 3：Transformer 三件套
- 自注意力
- 多头注意力
- 位置编码

## 必背 4：为什么主流是 Decoder-only
- 自回归生成统一
- 训练范式简单
- 通用生成能力强

## 必背 5：与模型交互的三个现实问题
- Prompt
- 采样参数
- Tokenizer / token 成本

## 必背 6：大模型局限
- 幻觉
- 知识过时
- 偏见
- 不确定性不足

## 必背 7：缓解幻觉的现代方案
- RAG
- 多步推理
- 外部工具调用

---

# 十五、一句话总结全章

> 大语言模型的发展，是语言建模从“统计计数”走向“向量表示”，再走向“注意力驱动的大规模自回归生成”的过程；而在今天真正可用的智能体系统里，最重要的已经不只是模型本身，而是模型与 Prompt、Tokenizer、检索、工具和工作流的协同。