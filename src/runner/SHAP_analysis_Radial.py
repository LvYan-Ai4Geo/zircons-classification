import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import shap
from src.config.config import PROCESSED_DIR, MODEL_DIR


# 1 配色
COLORS = ["#8CB2CF", "#D0DCEF", "#F6A8A1", "#D63E51"]
cmap = mcolors.LinearSegmentedColormap.from_list(
    "nature_style", COLORS
)
# 字体设置
plt.rcParams["font.family"] = "Times New Roman"
def draw_shap_radial_bar(data):

    labels = [d["id"] for d in data]
    values = np.array([d["val"] for d in data])

    # 1. 优化图形数值显示
    values = np.sqrt(values)   # ⭐核心：压缩极端值
    values = values / values.max()

    # 2️.角度
    N = len(values)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # 3️.每个柱不同颜色
    color_positions = np.linspace(0.1, 0.9, N)  # 避免极端浅/深
    colors = [cmap(p) for p in color_positions]

    # 4.画布
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = plt.subplot(111, polar=True)

    # 5.径向块
    bars = ax.bar(
        angles,
        values,
        width=2 * np.pi / N * 0.75,
        bottom=0.15,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9
    )

    # 6.标签优化
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)

    for label, angle in zip(ax.get_xticklabels(), angles):
        angle_deg = np.degrees(angle)
        if 90 < angle_deg < 270:
            label.set_rotation(angle_deg + 180)
        else:
            label.set_rotation(angle_deg)
        label.set_horizontalalignment('center')

    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["polar"].set_visible(False)
    plt.tight_layout()
    plt.savefig('Feature_SHAP_RadialBar', dpi=600, bbox_inches='tight')
    plt.show()

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
    model_file = MODEL_DIR / "best_rf_model_fea_dis_move_pca.pkl"
    pipeline = joblib.load(model_file)
    best_model = pipeline.steps[-1][1]

    # 3.可解释性分析
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)

    # 4.多分类SHAP处理
    mean_shap = np.abs(shap_values).mean(axis=2).mean(axis=0)
    total_importance = mean_shap.sum()
    analysis_data = []
    for i in range(len(feature_names)):

        analysis_data.append({
            "id": feature_names[i],
            "val": mean_shap[i],
            "pct": mean_shap[i] / total_importance * 100
        })

    # 5.Top15
    analysis_data = sorted(
        analysis_data,
        key=lambda x: x["val"],
        reverse=True
    )[:15]

    for i in range(len(analysis_data)):
        print(f'id:{analysis_data[i]['id']},value:{analysis_data[i]['val']}')

    # 绘图
    draw_shap_radial_bar(analysis_data)