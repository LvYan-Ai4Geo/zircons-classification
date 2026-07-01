import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import pandas as pd
import shap
from src.config.config import PROCESSED_DIR, MODEL_DIR

# 配色 (保持原样)
COLORS = ["#8CB2CF", "#D0DCEF", "#F6A8A1", "#D63E51"]
cmap = mcolors.LinearSegmentedColormap.from_list(
    "nature_style", COLORS
)
# 字体设置 (保持原样)
plt.rcParams["font.family"] = "Times New Roman"


def draw_shap_importance_chart(data):
    ids = [d["id"] for d in data]
    values = np.array([d["val"] for d in data])
    # pcts = np.array([d["pct"] for d in data])

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    colors = cmap(norm(values))

    # 1. 画布布局调整：分为3列 [网络图, 条形图, 色标]
    # 宽度比例调整：网络图(1.2) + 条形图(4) + 色标(0.7)
    fig = plt.figure(figsize=(15, 7), facecolor="white")
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.2, 4, 0.7],
        wspace=0.05
    )

    ax_net = fig.add_subplot(gs[0])  # 新增：网络图
    ax_bar = fig.add_subplot(gs[1])  # 原：条形图
    ax_legend = fig.add_subplot(gs[2])  # 原：色标

    # ==========================================
    # 2. 绘制左侧 TOP 15 汇聚网络图 (ax_net)
    # ==========================================
    ax_net.axis("off")  # 隐藏坐标轴

    y_pos = np.arange(len(ids))
    # 设置 Y 轴范围与条形图一致，确保连线对齐
    # 注意：这里设置 ylim 后，下面必须 invert_yaxis 才能与 ax_bar 对应
    ax_net.set_ylim(-0.5, len(ids) - 0.5)
    ax_net.invert_yaxis() # 【修正1】：翻转Y轴，使 Index 0 (Ce) 在顶部，与 ax_bar 一致
    ax_net.set_xlim(0, 10)  # 内部坐标系

    # 绘制 "TOP 15" 圆圈
    # 位置：X=2, Y=中间
    center_x, center_y = 2, len(ids) / 2 - 0.5
    circle = patches.Circle((center_x, center_y), radius=1.2,
                            facecolor='#3B6CB5', edgecolor='white', linewidth=2, zorder=10)
    ax_net.add_patch(circle)
    ax_net.text(center_x, center_y, "TOP\n15", color='white',
                ha='center', va='center', fontsize=14, fontweight='bold', zorder=11)

    # 绘制连接线 (从圆圈右侧发散到右侧边界)
    start_x_edge = center_x + 1.2
    # 【修正2】：减小 end_x，避免线条延伸到 ax_bar 的标签处造成遮挡
    # 原来 9.8 太靠右了，现在设为 8.5，留出右侧空间
    end_x = 8.5

    for i in range(len(ids)):
        y_val = i
        color = cmap(norm(values[i]))
        # 线宽映射：Value越大越粗，范围 1.5 到 5
        lw = 1.5 + 3.5 * norm(values[i])

        # 使用贝塞尔曲线绘制平滑连接线
        # 控制点设计：先水平向右，再弯曲到目标 Y
        # 调整控制点以适应新的 end_x (8.5)
        verts = [
            (start_x_edge, center_y),  # 起点 (圆圈边缘)
            (4, center_y),  # 控制点1
            (6.5, y_val),  # 控制点2 (稍微提前弯曲)
            (end_x, y_val),  # 终点
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                                  linewidth=lw, zorder=5)
        ax_net.add_patch(patch)

    # ==========================================
    # 3. 绘制中间横向条形图 (ax_bar) - 保持原代码逻辑
    # ==========================================
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

    # ==========================================
    # 4. 颜色标尺 (ax_legend) - 保持原代码逻辑
    # ==========================================
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

    # plt.savefig('Feature SHAP Network_Bar.png', dpi=600, bbox_inches='tight')
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
    model_file = MODEL_DIR / "best_xgb_model_fea_dis_move_pca_0.3.pkl"
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

    # 5. Top 15 (已修改为15)
    analysis_data = sorted(
        analysis_data,
        key=lambda x: x["val"],
        reverse=True
    )[:15]

    for i in range(len(analysis_data)):
        # 修复 f-string 中的引号问题
        print(f'id: {analysis_data[i]["id"]}, value: {analysis_data[i]["val"]}')

    # 绘图
    draw_shap_importance_chart(analysis_data)


