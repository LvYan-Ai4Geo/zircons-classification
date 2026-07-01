import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.lines import Line2D

from src.config.config import RAW_DIR


def sample_view(file_path):
    """
    :param file_path: 数据路径
    :return: 样品点位图
    """
    # 1. 读取数据
    df = pd.read_csv(file_path, header=0).dropna()

    # 2. 全局字体设置（Times New Roman）
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    # 3. 创建画布与投影
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # 4. 海洋渐变背景
    lon = np.linspace(-180, 180, 720)
    lat = np.linspace(-90, 90, 360)
    lon2d, lat2d = np.meshgrid(lon, lat)

    ocean_gradient = np.abs(lat2d) / 90.0

    ax.imshow(
        ocean_gradient,
        extent=[-180, 180, -90, 90],
        origin="lower",
        cmap=plt.cm.Blues,
        alpha=0.75,
        transform=ccrs.PlateCarree(),
        zorder=0
    )

    # 4. 陆地地形阴影
    ax.stock_img()
    # 降饱和遮罩
    ax.add_patch(
        plt.Rectangle(
            (-180, -90), 360, 180,
            facecolor="white",
            alpha=0.18,
            transform=ccrs.PlateCarree(),
            zorder=2
        )
    )

    # 5. 样品点
    sample_scatter = ax.scatter(
        df["lon"],
        df["lat"],
        marker="*",
        s=30,
        c="darkred",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=10
    )

    # 6. 图例（Legend）
    legend_elements = [
        Line2D(
            [0], [0],
            marker="*",
            color="w",
            label="Zircon samples",
            markerfacecolor="darkred",
            markeredgecolor="black",
            markersize=10
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.05),
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=11
    )

    # 7. 网格线
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        linestyle="--",
        color="gray",
        alpha=0.5
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    # ==============================
    # 比例尺与指北针
    # ==============================
    add_scalebar(ax, length_km=6000, segments=2)
    add_north_arrow(ax)

    # 9. 输出
    plt.savefig(
        "global_zircon_distribution_top_tier.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()


def add_scalebar(ax, length_km=6000, segments=2):
    bar_width = 0.18
    bar_height = 0.015

    x_start = 1 - bar_width - 0.05
    y_start = 0.08

    segment_width = bar_width / segments

    for i in range(segments):
        color = "black" if i % 2 == 0 else "white"
        rect = plt.Rectangle(
            (x_start + i * segment_width, y_start),
            segment_width,
            bar_height,
            facecolor=color,
            edgecolor="black",
            transform=ax.transAxes,
            zorder=100
        )
        ax.add_patch(rect)

    border = plt.Rectangle(
        (x_start, y_start),
        bar_width,
        bar_height,
        fill=False,
        edgecolor="black",
        linewidth=1,
        transform=ax.transAxes,
        zorder=101
    )
    ax.add_patch(border)

    ax.text(x_start,
            y_start + bar_height + 0.01,
            "0",
            transform=ax.transAxes,
            fontsize=11,
            ha="center")

    ax.text(x_start + bar_width / 2,
            y_start + bar_height + 0.01,
            f"{int(length_km / 2):,}",
            transform=ax.transAxes,
            fontsize=11,
            ha="center")

    ax.text(x_start + bar_width,
            y_start + bar_height + 0.01,
            f"{length_km:,} km",
            transform=ax.transAxes,
            fontsize=11,
            ha="center")


def add_north_arrow(ax):
    """
    在比例尺正上方居中放置指北针（箭头 + N 标注）
    使用 ax.annotate 绘制，纯 transAxes 坐标，兼容所有 matplotlib 版本
    """
    # 与 add_scalebar 中完全一致的参数
    bar_width = 0.18
    x_start = 1 - bar_width - 0.05
    x_center = x_start + bar_width / 2

    # 比例尺最高点 = y_start(0.08) + bar_height(0.015) + 数字间距(0.01) ≈ 0.105
    # 指北针底部 = 比例尺最高点 + 额外留白(0.025)
    arrow_bottom_y = 0.13

    # ---- 1) 画指北箭头 ----
    ax.annotate(
        "",                                                  # 不显示文字
        xy=(x_center, arrow_bottom_y + 0.05),               # 箭头尖端（顶部）
        xytext=(x_center, arrow_bottom_y),                   # 箭头尾部（底部）
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->,head_width=0.4,head_length=0.25", # 经典三角箭头
            lw=2,
            color="black"
        ),
        zorder=200
    )

    # ---- 2) 画 "N" 文字（箭头顶部） ----
    ax.text(
        x_center,
        arrow_bottom_y + 0.065,
        "N",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=201
    )


if __name__ == '__main__':
    sample_view(RAW_DIR / 'BiShe-total_data.CSV')
