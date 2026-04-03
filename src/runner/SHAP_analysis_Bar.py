import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import shap
from src.config.config import PROCESSED_DIR, MODEL_DIR


# 配色
COLORS = ["#8CB2CF", "#D0DCEF", "#F6A8A1", "#D63E51"]
cmap = mcolors.LinearSegmentedColormap.from_list(
    "nature_style", COLORS
)
# 字体设置
plt.rcParams["font.family"] = "Times New Roman"


def draw_shap_importance_chart(data):
    ids = [d["id"] for d in data]
    values = np.array([d["val"] for d in data])
    pcts = np.array([d["pct"] for d in data])

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    colors = cmap(norm(values))

    # 1.画布
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.5, 0.7],
        wspace=0.02
    )
    ax_bar = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    # 2.横向条形图
    y_pos = np.arange(len(ids))
    ax_bar.barh(
        y_pos,
        values,
        color=colors,
        height=0.6,
        zorder=2
    )

    ax_bar.invert_yaxis()
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(ids, fontsize=12)
    ax_bar.set_xlabel(
        "SHAP Importance (Mean |SHAP value|)",
        fontsize=14,
    )

    ax_bar.set_xlim(0, values.max() * 1.1)
    ax_bar.tick_params(direction="in")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for spine in ax_bar.spines.values():
        spine.set_linewidth(0.8)

    # 3.颜色标尺
    ax_legend.axis("off")
    gradient = np.linspace(1, 0, 256).reshape(-1, 1)
    ax_cbar = ax_legend.inset_axes([-0.1, 0.05, 0.18, 0.9])
    ax_cbar.imshow(
        gradient,
        aspect="auto",
        cmap=cmap
    )
    ax_cbar.axis("off")
    ax_legend.text(
        0.18,
        0.96,
        "High Contribution",
        fontsize=12,
        va="center"
    )
    ax_legend.text(
        0.18,
        0.04,
        "Low Contribution",
        fontsize=12,
        va="center"
    )
    plt.savefig('Feature SHAP Bar', dpi=600, bbox_inches='tight')
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


    # 4.多分类 SHAP 处理
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
    )[:22]

    for i in range(len(analysis_data)):
        print(f'id:{analysis_data[i]['id']},value:{analysis_data[i]['val']}')
    # 绘图
    draw_shap_importance_chart(analysis_data)


