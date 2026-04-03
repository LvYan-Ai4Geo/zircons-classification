import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
# 我自定义的配置文件,主要是一些路径
from src.config.config import RAW_DIR, MODEL_DIR, PROCESSED_DIR


def train(random_state,model_path,x_train,y_train,std_or_rb = None,use_smote = False):
    '''
    :param random_state: 随机数种子
    :param model_path: 模型保存路径
    :param x_train: 训练集数据
    :param y_train: 训练集标签
    :param std_or_rb: 标准化方法选择: Standard or Robust
    :param use_smote: 是否过采样
    :return:None
    '''

    # ==============================
    # 1. 是否使用smote过采样
    # ==============================
    if use_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
        print(f'SMOTE过采样: {len(x_train)} -> {len(x_train_resampled)} 样本')
    else:
        x_train_resampled, y_train_resampled = x_train, y_train


    # ==============================
    # 2. 构建Pipeline (RandomForest)
    # ==============================
    pipeline = Pipeline([
        ('pca',PCA(n_components=None)),
        ('scaler', std_or_rb),
        ('rf', RandomForestClassifier(
            n_jobs=-1,               # 使用所有CPU核心
            class_weight='balanced',
            random_state=random_state,
        ))
    ])

    # ==============================
    # 3. 贝叶斯搜索空间 (针对随机森林调优)
    # ==============================
    # 获取 Pipeline 步骤名称，防止手写前缀出错
    model_step_name = 'rf'

    search_spaces = {
        f'{model_step_name}__n_estimators': Integer(100, 350),      # 树的数量
        f'{model_step_name}__max_depth': Integer(10, 25),           # 随机森林通常需要较深的树
        f'{model_step_name}__min_samples_split': Integer(2, 10),    # 分裂所需最小样本数
        f'{model_step_name}__min_samples_leaf': Integer(2, 10),     # 叶子节点最小样本数
        f'{model_step_name}__max_features': Real(0.4, 1.0),         # 特征采样比例
        f'{model_step_name}__max_samples': Real(0.6, 0.9)           # 样本采样比例（bootstrap样本数）
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    opt = BayesSearchCV(
        pipeline,
        search_spaces,
        n_iter=40,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,           # 并行搜索
        verbose=0,
        random_state=42
    )

    # 4. 训练
    print('模型训练中')
    opt.fit(
        x_train_resampled,
        y_train_resampled,
    )

    current_best_model = opt.best_estimator_
    # 训练集指标
    y_train_pred = current_best_model.predict(x_train_resampled)
    train_acc = accuracy_score(y_train_resampled, y_train_pred)

    tqdm.write(
        f'BEST CV F1:{opt.best_score_:.4f}, '
        f'TRAIN ACC:{train_acc:.4f}, '
        f'BEST PARAMS:{opt.best_params_}'
    )

    print('模型优化，保存模型')
    best_model = current_best_model
    joblib.dump(best_model,model_path) # 保存模型


if __name__ == '__main__':

    # 1. 指定数据
    x_train = pd.read_csv(PROCESSED_DIR / 'x_train_fea_move.csv').iloc[:,1:]
    # print(x_train.shape)
    # print(x_train)
    y_train = pd.read_csv(PROCESSED_DIR / 'y_train_fea_move.csv').iloc[:, 1].values.ravel()

    # 指定当前模型的保存路径
    model_path = MODEL_DIR / 'best_rf_model_fea_dis_move_extract_pca.pkl'
    random_state = 42
    rb = RobustScaler()
    std = StandardScaler()
    train(random_state,model_path, x_train, y_train,std_or_rb=std,use_smote=False)
