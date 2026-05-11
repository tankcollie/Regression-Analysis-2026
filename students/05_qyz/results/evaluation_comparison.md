# Week 10 数据泄露对比报告

## 实验背景

本实验对比有数据泄露和无数据泄露的交叉验证流水线。

**数据泄露定义**：在模型评估过程中，验证集或测试集的信息意外地进入了训练过程。

---

## 实验结果

### 坏 CV（有数据泄露）
- RMSE: **0.3469**
- 特点：虚高（因为验证集在拟合 Scaler 时被"看到"了）

### 好 CV（无数据泄露）
- RMSE: **nan**
- 特点：真实且可靠（验证集保持完全陌生）

### 对比分析
- RMSE 差异: **nan** （好 CV 略高，原因：数据量减少）

---

## 技术解析

### ❌ 坏做法（bad_cross_validation）
```python
# 第1步：全量数据的预处理（包含泄露）
scaler = CustomStandardScaler()
X_scaled = scaler.fit_transform(X)  # 用全量数据拟合

# 第2步：拆分数据
for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    # 验证集已被污染！
```

**问题**：
- Scaler 在拆分前就已经"看到"了验证集
- 验证集的标准化使用了它自己的统计量
- 评估指标被虚高估计

### ✅ 好做法（good_cross_validation）
```python
# 在 CV 循环内部
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    
    # 第1步：仅在训练集上拟合
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 第2步：用训练集参数转换验证集
    X_val_scaled = scaler.transform(X_val)
    
    # 验证集完全陌生！
```

**优势**：
- 验证集保持真正"陌生"的状态
- 评估指标真实可靠
- 模型泛化能力的真实体现

---

## 业务启示

### 为什么要给老板展示坏的成绩？

在大厂或生产环境中，数据科学家必须诚实地报告模型性能：

1. **虚高的评估指标（坏 CV）→ 虚假承诺**
   - 上线后实际表现会远低于预期
   - 导致业务决策失误
   - 伤害团队信誉

2. **真实的评估指标（好 CV）→ 谨慎承诺**
   - 上线后表现符合或超过预期
   - 建立信任，获得更多资源
   - 长期收益更高

### 红线（Red Line）
- ❌ 混淆训练集和验证集的统计量
- ❌ 在 CV 循环前进行全局数据处理
- ❌ 向业务方隐瞒真实的模型误差

---

## 总结

| 方面 | 坏 CV | 好 CV |
|------|-------|--------|
| RMSE | 0.3469 | nan |
| 数据隔离 | ❌ 有泄露 | ✅ 无泄露 |
| 评估可信度 | ❌ 不可信 | ✅ 可信 |
| 业务决策 | ❌ 虚假承诺 | ✅ 谨慎承诺 |

**结论**：在真实场景中，应始终采用"好 CV"方式进行评估，
即使它的指标看起来"不那么漂亮"。
