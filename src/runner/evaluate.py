import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.config.config import RAW_DIR, MODEL_DIR, PROCESSED_DIR


def evaluate(x_data, y_data, model_file, label_mapping=None):
    """
    评估模型性能，包含混淆矩阵可视化
    参数:
    x_data: 特征数据
    y_data: 真实标签
    model_file: 模型文件路径
    label_mapping: 标签映射字典，例如 {0: '类别A', 1: '类别B', 2: '类别C', 3: '类别D'}
    """

    # ========== 基础设置 ==========
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['axes.unicode_minus'] = False

    # 模型加载
    model = joblib.load(model_file)
    print('----- 模型加载成功 -----')
    # 模型预测
    prediction = model.predict(x_data)
    accu = accuracy_score(y_data, prediction)
    f1 = f1_score(y_data, prediction, average='macro')
    recall = recall_score(y_data, prediction, average='macro')
    precision = precision_score(y_data, prediction, average='macro')

    print(f'准确率：{accu:4f}')
    print(f'精确率：{precision:4f}')
    print(f'召回率：{recall:4f}')
    print(f'f1_score：{f1:4f}')

    # 计算混淆矩阵（比例形式）
    cm = confusion_matrix(y_data, prediction, normalize='true')  # 使用归一化，得到比例

    print("\n----- 混淆矩阵（比例）-----")
    print(np.round(cm, 3))  # 打印4位小数的比例值

    # 获取标签名称
    if label_mapping is not None:
        labels = [label_mapping[i] for i in range(len(label_mapping))]
    else:
        labels = [str(i) for i in range(len(np.unique(y_data)))]

    # 科研级配色 - Nature/Science风格
    cmap = sns.light_palette("#3B6FB6", as_cmap=True)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%',
        cmap=cmap,
        vmin=0,
        vmax=1,  # ⭐关键：锁死颜色范围
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 12}
    )

    # plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20) # 取消了标题，不需要
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # 调整刻度标签
    plt.xticks(fontsize=10, rotation=0, ha='center')
    plt.yticks(fontsize=10)

    plt.tight_layout()
    # plt.savefig('svm_confusion_matrix_extract', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    x_data = pd.read_csv(PROCESSED_DIR / 'x_test_fea_move.csv').iloc[:,1:]
    y_data = pd.read_csv(PROCESSED_DIR / 'y_test_fea_move.csv').iloc[:,1:]

    model_file = MODEL_DIR / 'best_rf_model_fea_dis_move_extract_pca_rffea.pkl'
    # model = joblib.load(model_file)
    # print(model)
    label = {0:'Detrital',1:'Hydrothermal',2:'Magmatic',3:'Metamorphic'}
    evaluate(x_data,y_data,model_file,label_mapping=label)
