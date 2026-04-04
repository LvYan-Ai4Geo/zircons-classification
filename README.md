# 这是一个毕设项目，锆石的多分类任务。
## 1.src目录：前期进行了一些数据探索，比如点位图的绘制、数据的分布形态
在src目录下的data_preorocess目录下,lat_lon_view脚本是点位图绘制脚本源码，其他的一些ipy文件涉及到数据的初期处理与保存，包括boxplot进阶版箱线图的绘制、PCA分析、correlation相关性分析及一些特征构造的处理，具体进入源码脚本查看。
src目录下的config目录是我自己的一些路径配置文件。
## 2. model目录：保存的是训练过的各类机器学习模型。
## 3. runner目录：模型训练与评估的主目录
### 1. random_foreast_trainer脚本是随机森林模型训练的脚本，其中通过pipeline一体化流水线进行模型训练，结合贝叶斯优化、交叉验证，以f1_macro为评估指标，进行模型的调优
### 2. svm_trainer同理对应相应模型的训练脚本
### 3. xgb_trainer同上
### 4. evaluate脚本是模型评估脚本，使用了分类任务的四大指标:accuracy、pre、recall、f1,以及混淆矩阵的绘制
### 5. SHAP_analysi脚本是特征重要性分析脚本，衍生的一些蜂巢图、雷达径块图对应相应名称的脚本
