# ECG AIWIN

2021AIWIN（秋）——心电图智能诊断竞赛

### Software

- Python 3.7.5
- PyTorch 1.7.0

## Run

（1）通过resnet模型提取特征，然后利用提取的特征训练lgb模型，然后进行多标签预测。最后对预测的结果进行校正，如果结果中即有normal和剩下的11种病，则通过一个二分类模型进行校正。

（2）多标签分类模型——先训练resnet模型：train_resnet_1d_normal.py， train_resnet_2d_normal.py，再训练lgb模型：train_lgb_2_model_normal.py

（3）二分类模型——先训练resnet模型：然后使用训练好的resnet模型抽取训练集的特征，保存在feature_train.csv文件中，使用抽取的特征训练lightgbm模型：train_lgb.py 

（4）使用多标签分类模型和二分类模型进行预测：predict_answer.py
