# Gait-Recogonize

Use Machine Learning Algrithoms to analysis people's gait and recogonize them by walking video.

# 基于GEI的HOG特征，使用SVM分类器分类步态识别 
采用中科院自动化所GaitDatasetA-silh小型步态识别数据集进行实验分析

将数据集每人序列帧图像平均分为两部分，一部分为训练集，一部分为测试集

对行人二值化序列帧图像求平均得到GEI步态能量图，再采用OpenCV内置HOGDescriptor提取出GEI的HOG特征值，利用HOG特征训练SVM分类器，最终采用模型进行预测。识别率达到99.1667%
