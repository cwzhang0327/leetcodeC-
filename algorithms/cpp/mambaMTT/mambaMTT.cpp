// Date   : 2026-03-10

/***********************************************************************************
 *
 * 【中文翻译】
 *
 * 其中，W_{F_1} ∈ ℝ^{d×(e·d)} 和 W_{F_2} ∈ ℝ^{(e·d)×d} 是线性变换的权重矩阵。
 * Sigmoid(·) 是 Sigmoid 激活函数。Tanh(·) 是 Tanh 激活函数。
 *
 * 最终，O_{Mamba} 通过简单的线性变换被重新映射回原始特征维度空间，
 * O_{Mamba} 与最终输出 R 之间的关系可以表示为：
 *
 *   R = O_{Mamba} * W_R + b_R                               （公式 23）
 *
 * 其中，W_R ∈ ℝ^{d×4}，b_R ∈ ℝ^4。
 *
 * 由于本文采用了残差训练方法，需要将网络的输入序列与输出序列相结合，
 * 以确定真实轨迹的状态输出，记为 Z̄ = R + Z。
 *
 * 本文使用均方根误差（RMSE）作为 MambaMTT 网络的训练损失函数：
 *
 *           ┌──────────────────────────────
 *           │  1   M
 *   l  =   \│ ─── Σ  ( Z̄_k - X_k )²                       （公式 24）
 *           │  M  k=1
 *
 * 其中，M 为样本总数，Z̄_k 为第 k 个预测状态输出，X_k 为第 k 个真实值。
 *
 ***********************************************************************************/

#include <cmath>
#include <vector>
#include <stdexcept>

// Sigmoid 激活函数：σ(x) = 1 / (1 + e^{-x})
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Tanh 激活函数：tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
double tanh_activation(double x) {
    return std::tanh(x);
}

// 输出维度：W_R ∈ ℝ^{d×4}，b_R ∈ ℝ^4（论文固定值）
constexpr int OUTPUT_DIM = 4;

// 线性变换（公式 23）：R = O_Mamba * W_R + b_R
// 输入:
//   O_Mamba — 形状为 [1 × d] 的行向量
//   W_R     — 形状为 [d × OUTPUT_DIM] 的权重矩阵（按行展开）
//   b_R     — 长度为 OUTPUT_DIM 的偏置向量
// 输出:
//   R       — 长度为 OUTPUT_DIM 的输出向量
std::vector<double> linearTransform(
    const std::vector<double>& O_Mamba,
    const std::vector<std::vector<double>>& W_R,
    const std::vector<double>& b_R)
{
    int d = static_cast<int>(O_Mamba.size());
    if (static_cast<int>(W_R.size()) != d)
        throw std::invalid_argument("W_R 行数必须等于 O_Mamba 的维度 d");
    for (const auto& row : W_R)
        if (static_cast<int>(row.size()) != OUTPUT_DIM)
            throw std::invalid_argument("W_R 每行必须有 OUTPUT_DIM 列");
    if (static_cast<int>(b_R.size()) != OUTPUT_DIM)
        throw std::invalid_argument("b_R 长度必须为 OUTPUT_DIM");

    std::vector<double> R(OUTPUT_DIM, 0.0);
    for (int j = 0; j < OUTPUT_DIM; ++j) {
        for (int i = 0; i < d; ++i)
            R[j] += O_Mamba[i] * W_R[i][j];
        R[j] += b_R[j];
    }
    return R;
}

// 残差叠加：Z̄ = R + Z
std::vector<double> residualAdd(
    const std::vector<double>& R,
    const std::vector<double>& Z)
{
    if (R.size() != Z.size())
        throw std::invalid_argument("R 与 Z 的长度必须相同");

    std::vector<double> Z_bar(R.size());
    for (size_t i = 0; i < R.size(); ++i)
        Z_bar[i] = R[i] + Z[i];
    return Z_bar;
}

// 均方根误差损失函数（公式 24）：l = sqrt( (1/M) * Σ (Z̄_k - X_k)² )
// 输入:
//   Z_bar — 预测状态输出序列，长度为 M
//   X     — 真实值序列，长度为 M
// 输出:
//   RMSE 损失值
double rmseLoss(
    const std::vector<double>& Z_bar,
    const std::vector<double>& X)
{
    if (Z_bar.size() != X.size() || Z_bar.empty())
        throw std::invalid_argument("Z_bar 与 X 的长度必须相同且不为空");

    double sum = 0.0;
    int M = static_cast<int>(Z_bar.size());
    for (int k = 0; k < M; ++k) {
        double diff = Z_bar[k] - X[k];
        sum += diff * diff;
    }
    return std::sqrt(sum / M);
}
