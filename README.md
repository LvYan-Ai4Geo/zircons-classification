# 基于多种机器学习方法的锆石成因分类

> 本科毕业设计项目 —— 综合运用 **3 类机器学习模型 × 11 种特征工程组合**，对锆石样本进行成因分类（沉积型 / 热液型 / 岩浆型 / 变质型），并借助 **SHAP 可解释性分析**进行特征提取，最终得到最优分类模型。

---

## 一、研究背景

锆石（Zircon）是地质学中重要的示踪矿物，其微量元素特征能够反映母岩成因类型。传统判别方法多依赖人工经验与二元判别图解，存在主观性强、边界样本难界定等问题。本项目构建了一套数据驱动的锆石成因自动分类框架，通过系统比较多种特征工程与建模策略，探索最优分类方案，并利用 SHAP 揭示驱动分类的关键地球化学特征。

## 二、数据说明

### 2.1 数据来源

原始数据位于 `data/raw/BiShe-total_data.CSV`，共 **7218 条**锆石样本，每条记录包含经纬度与 13 种微量元素含量。

| 字段 | 说明 |
|------|------|
| `lat`, `lon` | 采样点经纬度 |
| `class` | 成因标签 |
| `Ce, Dy, Er, Eu, Gd, Ho, Lu, Nd, Sm, Th, Tm, U, Yb` | 13 种稀土/微量元素含量 (ppm) |

### 2.2 类别分布（存在类别不平衡）

| 类别 | 样本数 | 占比 |
|------|--------|------|
| Detrital（沉积型） | 3437 | 47.6% |
| Magmatic（岩浆型） | 2517 | 34.9% |
| Hydrothermal（热液型） | 536 | 7.4% |
| Metamorphic（变质型） | 400 | 5.5% |

> 少数类（变质型）与多数类比例约 **1 : 8.6**，是本项目引入 SMOTE 过采样与 `class_weight='balanced'` 的核心动机。

### 2.3 数据预处理

1. **缺失值处理**：剔除标签缺失样本；微量元素缺失值以中位数填充。
2. **数据划分**：按 `7 : 1.5 : 1.5` 分层划分（`stratify=y`, `random_state=42`）。
   - 训练集：4823 ｜ 验证集：1033 ｜ 测试集：1034
3. **标签编码**：`LabelEncoder`，仅在训练集 `fit`，映射关系为：
   `detrital→0, hydrothermal→1, magmatic→2, metamorphic→3`

## 三、特征工程

在 13 个原始稀土元素特征基础上，本项目设计了 **11 种特征工程组合**进行系统对比：

| 编号 | 缩写 | 组合说明 |
|------|------|----------|
| 1 | Rb | RobustScaler |
| 2 | Std | StandardScaler |
| 3 | Smo | SMOTE 过采样 |
| 4 | Fea | Feature Construction（特征构造） |
| 5 | Rb+Smo | RobustScaler + SMOTE |
| 6 | Std+Smo | StandardScaler + SMOTE |
| 7 | Fea+Rb | 特征构造 + RobustScaler |
| 8 | Fea+Std | 特征构造 + StandardScaler |
| 9 | Fea+Rb+Smo | 特征构造 + RobustScaler + SMOTE |
| 10 | Fea+Std+Smo | 特征构造 + StandardScaler + SMOTE |
| 11 | Fea+PCA | 特征构造 + PCA |
| 12 | Fea+PCA+Smo | 特征构造 + PCA + SMOTE |

### 特征构造（Feature Construction）

在 13 个原始元素基础上构造 **13 个衍生特征**，合计 **26 维**（见 `process_raw_feature.ipynb`）：

- **地质学比值特征（7 个）**
  - `LREE` = Ce + Nd + Sm + Eu（轻稀土总量）
  - `HREE` = Dy + Ho + Er + Yb + Lu + Tm（重稀土总量）
  - `LREE_HREE_ratio`（轻重稀土分异程度）
  - `Eu_anomaly` = Eu / √(Sm·Gd)（铕异常，关键地质指示指标）
  - `Nd_Yb_ratio`、`Th_U_ratio`、`Gd_Yb_ratio`
- **统计特征（6 个）**：`sum_REE, mean_REE, std_REE, max_REE, min_REE, range_REE`

### PCA 降维

对标准化后的特征进行主成分分析，保留 13 个主成分，消除特征间多重共线性（见 `process_raw_PCA.ipynb`）。

## 四、模型方法

采用 **3 类经典机器学习模型**，统一使用 **贝叶斯优化（BayesSearchCV）+ 10 折分层交叉验证**，以 `f1_macro` 为优化目标：

### 4.1 随机森林（Random Forest）
- 文件：`src/runner/random_foreast_trainer.py`
- 搜索空间：`n_estimators`, `max_depth`, `min_samples_split/leaf`, `max_features/samples`
- 内置 `class_weight='balanced'` 应对类别不平衡

### 4.2 支持持向量机（SVM）
- 文件：`src/runner/svm_trainer.py`
- 搜索空间：`C`（log-uniform 0.01~1，限制上限防小样本过拟合）、`kernel`（rbf/linear）
- `probability=True` 支持置信度输出，`decision_function_shape='ovr'`

### 4.3 XGBoost
- 文件：`src/runner/xgboost_trainer.py`
- 搜索空间：`n_estimators, max_depth, learning_rate, subsample, gamma, min_child_weight, reg_alpha, reg_lambda`
- 通过 `sample_weight` 注入类别权重以缓解不平衡

> 所有模型封装于 `sklearn.Pipeline`，将缩放器 / PCA / 分类器串联，保证训练-推理一致性。训练好的模型以 `.pkl` 形式保存于 `model/`。

## 五、可解释性分析（SHAP）

利用 **SHAP（SHapley Additive exPlanations）** 对最优模型进行特征归因，多分类场景下对类别维度取平均得到全局重要性：

| 脚本 | 输出 | 说明 |
|------|------|------|
| `SHAP_analysis_Bar.py` | `Feature SHAP Bar.png` | 横向条形图，展示 Top15 特征平均 \|SHAP\| |
| `SHAP_analysis_Bees.py` | `Feature SHAP Bees.png` | 蜂群图，展示特征值高低对模型输出的方向性影响 |
| `SHAP_analysis_Radial.py` | `Feature_SHAP_RadialBar.png` | 径向条形图，平方根压缩极端值后展示重要性分布 |

### 特征提取与最优模型

基于 SHAP 全局重要性，从 26 维构造特征中**提取 19 个关键特征子集**（`extract`）：

```
Ce, Sm, HREE, Dy, Gd, Nd, LREE, Th, Eu_anomaly, Yb, Gd_Yb_ratio,
Nd_Yb_ratio, Eu, Ho, Tm, Lu, U, sum_REE, Th_U_ratio
```

在该子集上重新训练（搭配 PCA + StandardScaler），得到最终最优模型：
`model/best_xgb_model_fea_dis_move_extract_pca.pkl`

## 六、项目结构

```
Bishe/
├── data/
│   ├── raw/                      # 原始数据与 train/valid/test 划分
│   │   ├── BiShe-total_data.CSV          # 主数据集（7218 样本）
│   │   ├── BiShe-total_data_CI.CSV       # CI 校正变体
│   │   └── x_{train,valid,test}_{raw,ci}.csv
│   └── processed/                # 特征工程产物
│       ├── *_fea.csv             # 含 log 特征的构造特征集
│       ├── *_fea_move.csv        # 最终采用的特征构造集（26 维）
│       └── *_PCA.csv             # PCA 降维特征集（13 维）
├── model/                        # 全部训练好的模型 (.pkl)
│   ├── best_rf_model_*.pkl       # 随机森林（多种特征工程组合）
│   ├── best_svm_model_*.pkl      # SVM
│   └── best_xgb_model_*.pkl      # XGBoost
├── src/
│   ├── config/config.py          # 路径配置
│   ├── data_preprocess/          # 数据分析与特征工程
│   │   ├── process_raw_feature.ipynb     # 特征构造 + 数据划分
│   │   ├── process_raw_PCA.ipynb         # PCA 降维
│   │   ├── process_raw_correlation.ipynb # 相关性分析
│   │   ├── data_analysis_boxplot.ipynb   # 箱线图 EDA
│   │   └── lat_lon_view.py               # 全球锆石采样点分布图
│   └── runner/                   # 模型训练与评估
│       ├── random_foreast_trainer.py
│       ├── svm_trainer.py
│       ├── xgboost_trainer.py
│       ├── evaluate.py           # 评估指标 + 混淆矩阵可视化
│       ├── SHAP_analysis_Bar.py
│       ├── SHAP_analysis_Bees.py
│       └── SHAP_analysis_Radial.py
└── README.md
```

## 七、复现指南

### 7.1 环境依赖

```
python ≥ 3.10
pandas, numpy, scikit-learn
imbalanced-learn          # SMOTE
xgboost
scikit-optimize (skopt)   # BayesSearchCV
shap
matplotlib, seaborn
cartopy                   # 地理可视化
joblib, tqdm
```

### 7.2 运行流程

```bash
# 1. 数据探索与特征工程（生成 data/processed/ 下的数据集）
#    依次执行 src/data_preprocess/ 下的各 notebook

# 2. 模型训练（在 *_trainer.py 的 __main__ 中指定数据与保存路径后运行）
python -m src.runner.random_foreast_trainer
python -m src.runner.svm_trainer
python -m src.runner.xgboost_trainer

# 3. 模型评估（输出 Accuracy / Precision / Recall / F1 + 混淆矩阵）
python -m src.runner.evaluate

# 4. SHAP 可解释性分析
python -m src.runner.SHAP_analysis_Bar
python -m src.runner.SHAP_analysis_Bees
python -m src.runner.SHAP_analysis_Radial
```

> 训练脚本采用参数化设计：通过 `__main__` 中的 `model_path`、`std_or_rb`、`use_smote` 三个参数即可切换不同特征工程组合，无需改动核心训练逻辑。

## 八、评估指标

针对多分类不平衡场景，采用**宏平均**指标以平等对待每个类别：

- **Accuracy**（准确率）
- **Precision / Recall / F1-score**（macro 平均）
- **混淆矩阵**（归一化比例形式，可视化各类别识别情况）

评估代码见 `src/runner/evaluate.py`，混淆矩阵采用 Nature/Science 风格配色，锁定 `[0,1]` 色域范围便于跨模型横向对比。

## 九、主要结论

1. **特征构造有效**：基于地球化学先验构造的比值与统计特征（尤其 `Eu_anomaly`、`Th_U_ratio`、`LREE_HREE_ratio`）显著提升了模型判别力。
2. **类别不平衡处理关键**：SMOTE 过采样与 `class_weight='balanced'` 双重策略显著改善了少数类（变质型、热液型）的召回率。
3. **SHAP 驱动的特征提取**：从 26 维特征中提取 19 维关键子集后，模型在降维的同时保持甚至提升了泛化性能，说明 SHAP 能够有效识别冗余特征。
4. **最优方案**：XGBoost + 特征构造 + 特征提取 + PCA + StandardScaler 组合取得最佳分类表现。
