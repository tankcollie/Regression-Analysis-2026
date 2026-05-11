# Week09 & Week 10 综合实验报告：数据清洗、模型诊断与防泄露流水线

## 实验概述

本报告涵盖两周的实验内容：

- **Week 9**: 数据急救与模型诊断 —— 实现 CLI 数据清洗脚本、One-Hot 编码、异常值缩尾处理、缺失值填补，以及 VIF 多重共线性诊断。
- **Week 10**: 工业流水线与无泄漏泛化评估 —— 实现 Transformer 接口的标准化器，对比有数据泄露和无数据泄露的交叉验证结果，分析数据泄露的危害。

**核心结论**：
1. 数据中存在严重多重共线性（TV_Budget、Online_Video_Budget VIF > 10）
2. 有数据泄露的交叉验证会严重低估泛化误差（MAPE 8.09% vs 13.73%）
3. 正确的预处理必须在交叉验证循环内部完成，只用训练集拟合统计量

---

## 第一部分：Week09 —— 数据急救与模型诊断

### 1.1 实验目的

现实世界的数据充满缺失值、异常值和多重共线性。本实验实现：
- 命令行数据清洗脚本（CLI）
- 多重共线性诊断（VIF）
- 基线模型交叉验证

### 1.2 数据清洗 CLI 脚本 (`src/week09/data_prep.py`)

#### 功能实现

| 功能 | 实现方式 | 作业要求 |
|------|----------|----------|
| 命令行参数 | `argparse` | 无硬编码路径 |
| 分类变量编码 | `pd.get_dummies(drop_first=True)` | 避免虚拟变量陷阱 |
| 异常值处理 | Winsorization（99分位数缩尾） | ✅ |
| 缺失值填补 | 全局均值填补 | 本周临时方案 |

#### 核心代码

```python
# One-Hot 编码（drop_first=True 是关键）
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 缩尾处理异常值
percentile_99 = df[col].quantile(0.99)
df[col] = np.where(df[col] > percentile_99, percentile_99, df[col])

# 均值填补缺失值
df = df.fillna(df.mean())

### 1.3 多重共线性诊断 (`src/utils/diagnostics.py`)

#### VIF 原理

$$VIF_j = \frac{1}{1 - R_j^2}$$

- **VIF = 1**：完全不相关
- **1 < VIF < 5**：中等相关，可接受
- **VIF ≥ 10**：严重多重共线性，需要处理

#### 核心代码

```python
def calculate_vif(X: np.ndarray) -> list:
    """计算每个特征的方差膨胀因子 (VIF)"""
    n_features = X.shape[1]
    vif_values = []
    
    for j in range(n_features):
        y_j = X[:, j]
        X_j = np.delete(X, j, axis=1)
        X_with_intercept = np.column_stack([np.ones(len(X_j)), X_j])
        
        model = AnalyticalOLS(fit_intercept=False)
        model.fit(X_with_intercept, y_j)
        
        y_pred = model.predict(X_with_intercept)
        sse = np.sum((y_j - y_pred) ** 2)
        sst = np.sum((y_j - np.mean(y_j)) ** 2)
        r_squared = 1 - sse / sst if sst > 0 else 0
        
        vif = 1 / (1 - r_squared) if r_squared < 0.999 else float('inf')
        vif_values.append(vif)
    
    return vif_values

## 第二部分：Week 10 —— 工业流水线与无泄漏泛化评估

### 2.1 实验目的

演示数据泄露的危害，构建绝对纯洁、无污染的交叉验证流水线。通过对比有数据泄露和无数据泄露的交叉验证结果，揭示为什么"好看"的指标可能是致命的陷阱。

### 2.2 扩充算法工具箱

#### 评估指标库 (`src/utils/metrics.py`)

| 指标 | 公式 | 业务含义 |
|------|------|----------|
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | 对大误差敏感，放大异常值影响 |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 对异常值不敏感，更稳健 |
| MAPE | $\frac{1}{n}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$ | 百分比误差，便于业务理解和沟通 |

#### 转换器 API (`src/utils/transformers.py`)

实现 `CustomStandardScaler` 类，严格遵循大厂 Transformer 接口规范：

| 方法 | 功能 | 说明 |
|------|------|------|
| `fit(X)` | 计算均值和标准差 | 只用训练集，保存到 `self.mean_` 和 `self.std_` |
| `transform(X)` | 标准化转换 | 用训练集的统计量转换数据 |
| `fit_transform(X)` | 拟合+转换 | 等价于 `fit(X).transform(X)` |

**关键设计**：`fit` 和 `transform` 分离，确保验证集只用 `transform`，不重新 `fit`。

### 2.3 数据泄露的危害对比实验

#### 实验设计

| 实验 | 预处理方式 | 数据流向 | 说明 |
|------|-----------|----------|------|
| **错误交叉验证** | 全局预处理 | 全量数据 → 标准化 → 填补 → 切分 | ❌ 验证集信息泄露 |
| **正确交叉验证** | 循环内预处理 | 切分 → 训练集 fit → 验证集 transform | ✅ 无数据泄露 |

#### 运行结果

| 指标 | 有数据泄露 (错误CV) | 无数据泄露 (正确CV) | 差异 |
|------|---------------------|---------------------|------|
| RMSE | 58.37 | 160.67 | -102.30 |
| MAE | 47.62 | 77.35 | -29.74 |
| **MAPE** | **8.09%** | **13.73%** | **-5.64%** |

#### 结果解读

**有数据泄露的版本各项指标都更"好看"**：

- MAPE 被低估了 5.64 个百分点（8.09% vs 13.73%）
- RMSE 被低估了 102 个单位
- MAE 被低估了 30 个单位

**但这恰恰是致命的陷阱**：模型在实际部署时，真实误差会比预期大得多！

### 2.4 可视化结果

![对比图](leakage_analysis.png)

- **红色柱**：有数据泄露（误差更小，但虚假）
- **绿色柱**：无数据泄露（真实泛化误差）
- **结论**：数据泄露会严重低估模型的实际误差

### 2.5 业务解读

模型上线后，每天的销售额预测平均绝对百分比误差约为 **13.73%**。

**业务含义**：
- 如果某天实际销售额为 **100 万元**，预测误差大约在 **±13.73 万元** 左右
- 这意味着每天的预算决策需要考虑约 14% 的不确定性
- 库存管理需要预留相应缓冲

**为什么老板应该看这个"差成绩"？**

| 看什么成绩 | 后果 |
|-----------|------|
| 泄露版成绩（8.09%） | 盲目乐观，备货不足，错失销售机会 |
| 真实成绩（13.73%） | 合理规划，稳健经营，风险可控 |

**核心教训**：在数据科学中，诚实的"差成绩"比虚假的"好成绩"更有价值。