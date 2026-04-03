import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import shap
from src.config.config import PROCESSED_DIR, MODEL_DIR


# 1 配色
COLORS = ["#8CB2CF", "#D0DCEF", "#F6A8A1", "#D63E51"]
# 蜂群图通常用于表示特征值高低，这里将配色映射为：
# 低特征值 -> 蓝色系 (#8CB2CF) | 高特征值 -> 红色系 (#D63E51)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "nature_style", COLORS
)
# 设置字体
plt.rcParams["font.family"] = "Times New Roman"



# 2 绘图函数 (蜂群图)
def draw_shap_beeswarm_chart(shap_values, features, feature_names):
    """
    绘制SHAP蜂群图
    :param shap_values: SHAP值矩阵 (n_samples, n_features)
    :param features: 特征值矩阵 (n_samples, n_features)，用于着色
    :param feature_names: 特征名称列表
    """

    # 1.使用shap原生绘图接口，但关闭自带显示以便自定义样式
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        plot_type="dot",  # dot 即为蜂群图
        cmap=cmap,  # 使用自定义配色
        show=False,  # 不立即显示，方便后续微调
        max_display=15  # 显示Top 15特征
    )

    # 2.坐标轴样式微调
    ax = plt.gca()
    # 2.1 调整坐标轴线宽和颜色
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)  # 调细
        spine.set_color("#555555")  # 调淡，不使用纯黑

    # 2.2 调整刻度样式
    ax.tick_params(
        axis="both",
        direction="in",
        colors="#333333",
        labelsize=12
    )

    # 2.3 调整X轴标签
    ax.set_xlabel("SHAP Value (Average impact on model output magnitude)", fontsize=14)
    plt.tight_layout()
    plt.savefig('Feature SHAP Bees', dpi=600, bbox_inches='tight')
    plt.show()


# ===============================
# 3 主程序
# ===============================

if __name__ == "__main__":
    # ========== 基础设置 ==========
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['axes.unicode_minus'] = False

    # 1.读取数据
    X = pd.read_csv(PROCESSED_DIR / "x_train_fea_move.csv").iloc[:, 1:]
    y = pd.read_csv(PROCESSED_DIR / "y_train_fea_move.csv").iloc[:, 1].values.ravel()
    feature_names = X.columns.tolist()

    # 2.加载模型
    model_file = MODEL_DIR / "best_xgb_model_fea_dis_move_pca_0.3.pkl"
    pipeline = joblib.load(model_file)
    best_model = pipeline.steps[-1][1]

    # 3.可解释性分析
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)

    # 4.数据预处理：多分类 -> 全局重要性
    # 原始 shap_values 形状: (n_samples, n_features, n_classes)
    # 蜂群图需要 2D 矩阵，因此对 '类别' 维度取平均
    # 得到形状: (n_samples, n_features)
    shap_values_2d = np.mean(shap_values, axis=2)

    # 5.特征值矩阵用于给点着色 (显示特征值高低)
    features_matrix = X.values


    # 6.筛选Top 15 特征 (保持原有排序逻辑)
    # 计算平均绝对SHAP值用于排序
    mean_shap_importance = np.abs(shap_values_2d).mean(axis=0)
    top_indices = np.argsort(mean_shap_importance)[::-1][:15]

    # 7.提取对应数据
    shap_top = shap_values_2d[:, top_indices]
    features_top = features_matrix[:, top_indices]
    names_top = [feature_names[i] for i in top_indices]

    # 8.绘图
    draw_shap_beeswarm_chart(shap_top, features_top, names_top)
