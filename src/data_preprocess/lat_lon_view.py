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
    df = pd.read_csv(file_path,header=0).dropna()
    # print(df)

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
    # 标准比例尺（右下角对齐图例）
    # ==============================


    add_scalebar(ax, length_km=6000, segments=2)

    # 9. 输出
    plt.savefig(
        "global_zircon_distribution_top_tier.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()


def add_scalebar(ax, length_km=6000, segments=2):
    # —— 关键：右下角位置 —— #
    bar_width = 0.18
    bar_height = 0.015

    x_start = 1 - bar_width - 0.05  # 右侧留边距
    y_start = 0.08  # 与图例基本水平

    segment_width = bar_width / segments

    # 黑白分段
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

    # 外边框
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

    # 数字
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

if __name__ == '__main__':
    sample_view(RAW_DIR / 'BiShe-total_data.CSV')