import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

from src.config.config import RAW_DIR, MODEL_DIR, PROCESSED_DIR
from skopt import BayesSearchCV
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from skopt.space import Real, Categorical


def train(random_state,model_path, x_train, y_train,rb_or_std = None,use_smote=False):

    # ==============================
    # 1.是否适用Smote过采样
    # ==============================
    if use_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
        print(f'SMOTE过采样: {len(x_train)} -> {len(x_train_resampled)} 样本')
    else:
        x_train_resampled, y_train_resampled = x_train, y_train


    # ==============================
    # 2.Pipeline 设置
    # ==============================
    # 小样本必开 probability=True 以便输出置信度分析
    pipeline = Pipeline([
        ('pca',PCA(n_components=None)),
        ('std',rb_or_std),
        ('svc', SVC(
            class_weight='balanced',
            probability=True,
            decision_function_shape='ovr',
            gamma='auto',
            random_state=random_state
        ))
    ])

    # ==============================
    # 3. 搜索空间 (针对小样本防过拟合调整)
    # ==============================
    # 关键调整：限制 C 的上限。
    # 小样本中过大的 C 会导致模型死记硬背少数类样本（过拟合）
    model_step_name = 'svc'
    search_spaces = {
        f'{model_step_name}__C': Real(0.01, 1, prior='log-uniform'),
        f'{model_step_name}__kernel': Categorical(['rbf', 'linear'])
    }

    # ==============================
    # 4. 交叉验证
    # ==============================
    # 关键调整：使用 10 折交叉验证。
    # 因为只有 4000+ 样本，每折数据量少，增加折数能更充分利用数据评估模型稳定性
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    opt = BayesSearchCV(
        pipeline,
        search_spaces,
        n_iter=40,  # 数据少跑得快，加到 50 次迭代找更优解
        cv=cv,  # 10折 CV
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0,
        random_state=random_state
    )

    # ==============================
    print('模型训练中')
    # 5. 执行训练
    opt.fit(x_train_resampled, y_train_resampled)
    best_model = opt.best_estimator_
    y_train_pred = best_model.predict(x_train_resampled)
    train_f1 = f1_score(y_train_resampled, y_train_pred, average='macro')
    cv_f1 = opt.best_score_

    # 6. 打印分类报告
    print(classification_report(y_train_resampled, y_train_pred, digits=4))

    # 保存模型
    joblib.dump(best_model, model_path)
    print(f'模型已保存至: {model_path}')
    return best_model


if __name__ == '__main__':

    # 1. 指定数据
    x_train = pd.read_csv(PROCESSED_DIR / 'x_train_fea_move.csv').iloc[:,1:]
    # print(x_train.shape)
    # print(x_train)
    y_train = pd.read_csv(PROCESSED_DIR / 'y_train_fea_move.csv').iloc[:, 1].values.ravel()

    # 指定当前模型的保存路径
    model_path = MODEL_DIR / 'best_svm_model_fea_dis_move_extract_pca.pkl'
    random_state = 42
    rb = RobustScaler()
    std = StandardScaler()
    train(random_state,model_path,x_train, y_train,rb_or_std=std,use_smote=False)

