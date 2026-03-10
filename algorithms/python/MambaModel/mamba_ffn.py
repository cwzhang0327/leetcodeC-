"""
Mamba 模型前馈网络模块（FFN）实现

原文说明（中文翻译）：
    其中，W_F1 ∈ R^{d × (e·d)}，W_F2 ∈ R^{(e·d) × d} 是线性变换的权重矩阵。
    Sigmoid(·) 是 Sigmoid 激活函数。
    Tanh(·) 是 Tanh 激活函数。
    最后，O_Mamba 通过一个简单的线性变换被重新映射回原始特征维度空间，
    O_Mamba 与最终输出 R 之间的关系可以表示如下：
        R = O_Mamba · W_out
    其中 W_out ∈ R^{d × d} 是输出线性变换的权重矩阵。
"""

import math


def sigmoid(x):
    """Sigmoid 激活函数：将输入压缩到 (0, 1) 区间（数值稳定实现）。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def tanh(x):
    """Tanh 激活函数：将输入压缩到 (-1, 1) 区间。"""
    return math.tanh(x)


def linear_transform(x, weight):
    """
    线性变换：对输入向量 x 应用权重矩阵 weight。

    参数:
        x (list[float])      : 输入向量，维度为 d_in。
        weight (list[list])  : 权重矩阵，形状为 (d_out, d_in)。

    返回:
        list[float]: 输出向量，维度为 d_out。

    异常:
        ValueError: 当权重矩阵的列数与输入向量维度不匹配时抛出。
    """
    d_out = len(weight)
    d_in = len(x)
    for i, row in enumerate(weight):
        if len(row) != d_in:
            raise ValueError(
                f"权重矩阵第 {i} 行的长度 ({len(row)}) 与输入向量维度 ({d_in}) 不匹配。"
            )
    result = []
    for i in range(d_out):
        val = sum(weight[i][j] * x[j] for j in range(d_in))
        result.append(val)
    return result


def mamba_ffn(x, W_F1, W_F2):
    """
    Mamba 前馈网络（FFN）模块。

    先将输入 x 通过 W_F1 投影到扩展维度 (e·d)，
    然后分别应用 Sigmoid 和 Tanh 激活函数并逐元素相乘（门控机制），
    最后通过 W_F2 投影回原始维度 d，得到 O_Mamba。

    参数:
        x    (list[float])       : 输入向量，维度为 d。
        W_F1 (list[list[float]]) : 第一层权重矩阵，形状为 (e·d, d)。
        W_F2 (list[list[float]]) : 第二层权重矩阵，形状为 (d, e·d)。

    返回:
        list[float]: O_Mamba，维度为 d。
    """
    # 线性投影：x -> 扩展空间 (e·d)
    hidden = linear_transform(x, W_F1)

    # 门控激活：Sigmoid(hidden) ⊙ Tanh(hidden)
    gated = [sigmoid(h) * tanh(h) for h in hidden]

    # 线性投影：扩展空间 (e·d) -> 原始维度 d，得到 O_Mamba
    o_mamba = linear_transform(gated, W_F2)

    return o_mamba


def remap_to_output(o_mamba, W_out):
    """
    将 O_Mamba 通过线性变换重新映射回原始特征维度空间，得到最终输出 R。

    O_Mamba 与最终输出 R 的关系：
        R = O_Mamba · W_out

    参数:
        o_mamba (list[float])      : Mamba 模块的输出，维度为 d。
        W_out   (list[list[float]]): 输出权重矩阵，形状为 (d, d)。

    返回:
        list[float]: 最终输出 R，维度为 d。
    """
    return linear_transform(o_mamba, W_out)
