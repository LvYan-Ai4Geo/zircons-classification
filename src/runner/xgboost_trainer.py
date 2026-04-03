from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline, Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
import joblib
from tqdm import tqdm
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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
    # 1. 是否使用Smote过采样
    # ==============================
    if use_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
        print(f'SMOTE过采样: {len(x_train)} -> {len(x_train_resampled)} 样本')
    else:
        x_train_resampled, y_train_resampled = x_train, y_train


    # ==============================
    # 2. 处理类别不平衡，XGBoost需要统计，没有内置class_weighted
    # ==============================
    # 统计类别分布（用于计算权重）
    unique_classes, class_counts = np.unique(y_train_resampled, return_counts=True)
    class_weights = len(y_train_resampled) / (len(unique_classes) * class_counts)  # 平衡权重公式
    # 将权重转为XGBoost可识别的格式
    sample_weights = np.array([class_weights[np.where(unique_classes == c)[0][0]] for c in y_train_resampled])


    # ==============================
    # 2. 构建Pipeline
    # ==============================

    pipeline = Pipeline([
        ('scaler', std_or_rb),
        ('xgb', XGBClassifier(
            objective='multi:softprob', # 多分类任务
            eval_metric='mlogloss',
            random_state=random_state
        ))
    ])

    # ==============================
    # 3. 贝叶斯搜索空间
    # ==============================
    search_spaces = {
        'xgb__n_estimators': Integer(100, 350),
        'xgb__max_depth': Integer(3, 11),
        'xgb__learning_rate': Real(0.01, 0.4, prior='log-uniform'),
        'xgb__subsample': Real(0.45, 0.98),
        'xgb__gamma': Real(0, 5),
        'xgb__min_child_weight': Integer(5, 12),
        'xgb__reg_alpha': Real(1, 8),
        'xgb__reg_lambda': Real(5, 15)
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    opt = BayesSearchCV(
        pipeline,
        search_spaces,
        n_iter=45,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0,
        random_state=random_state
    )


    # 4. 模型训练
    print('模型训练中')
    opt.fit(
        x_train_resampled,
        y_train_resampled,
        xgb__sample_weight=sample_weights
    )
    # 最优模型
    current_best_model = opt.best_estimator_

    # 训练集指标
    y_train_pred = current_best_model.predict(x_train_resampled)
    train_acc = accuracy_score(y_train_resampled, y_train_pred)

    tqdm.write(
        f'BEST CV F1:{opt.best_score_:.4f}, '
        f'TRAIN ACC:{train_acc:.4f},'
        f'BEST PARAMS:{opt.best_params_}'
    )


    print('保存模型')
    best_model = current_best_model
    joblib.dump(best_model,model_path)



if __name__ == '__main__':

    # 1. 指定数据
    x_train = pd.read_csv(PROCESSED_DIR / 'x_train_fea_move.csv').iloc[:,1:]

    print(x_train)
    # print(x_train.shape)
    # print(x_train)
    y_train = pd.read_csv(PROCESSED_DIR / 'y_train_fea_move.csv').iloc[:, 1].values.ravel()

    # 指定当前模型的保存路径
    random_state = 42
    rb = RobustScaler()
    std = StandardScaler()
    model_path = MODEL_DIR / 'best_xgb_model_fea_dis_move_extract.pkl'
    train(random_state,model_path, x_train, y_train,std_or_rb=std,use_smote=False)