# Gait-Recogonize
Use Machine Learning Algrithoms to analysis people's gait and recogonize them by walking video.

实验环境 AI Studio CPU

# 基于GEI的HOG特征，使用SVM分类器分类步态识别

采用中科院自动化所GaitDatasetA-silh小型步态识别数据集进行实验分析

将数据集每人序列帧图像平均分为两部分，每人序列帧图像一半作为训练集，一部分为测试集（不与训练集重合）

对行人二值化序列帧图像求平均得到GEI步态能量图，再采用OpenCV内置HOGDescriptor提取出GEI的HOG特征值，利用HOG特征训练SVM分类器，最终采用模型进行预测。当测试集每人取序列帧一半识别率达到100%

测试集中每人序列帧图像从1张到17张，识别率从17.9%到96.25%增长，当测试集中每人序列帧图像达到10张时，识别率超过84%，按照每秒25帧计算，只需0.4秒即可识别出人员身份

## 图片预处理
对人体区域进行裁剪，将人体对称轴放于图片中央。


```python
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
%matplotlib inline


def cut_img(img, T_H, T_W):
    # 获得最高点和最低点
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # 如果高比宽要大，用高计算 resize 的比例
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv.resize(img, (_t_w, T_H), interpolation=cv.INTER_CUBIC)
    # 获得人的对称轴
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img
```

# GEI步态能量图


```python
def get_GEI(imgs):
    GEI = (imgs[0]/255).astype("uint8")
    for img in imgs[1:]:
        GEI += (img/255).astype("uint8")
    GEI = GEI/len(imgs)

    return (GEI*255).astype("uint8")
```

# 取HOG特征
采用opencv封装的HOGDescriptor进行HOG特征计算

block大小为8 * 8
cell大小为8 * 8
每个block 取 9 个 bin


```python

SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
def get_Hog(img):
    winSize = (img.shape[1], img.shape[0])      # winSize = (64,64)
    blockSize = (8,8)                               # blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,
                  cellSize,nbins,derivAperture,
                  winSigma,histogramNormType,L2HysThreshold,
                  gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = [] # (10, 10)# ((10,20),)
    hist = hog.compute(img,winStride,padding,locations)
    # hist.dtype = np.float64
    return hist
```

# 读取GaitDatasetA-silh数据集文件


```python
roots = "/home/aistudio/GaitDatasetA-silh/"
cells = []
labels = []
label = os.listdir(roots)
print(label)
# print(label)
for dir_ in label:
    dirs = os.listdir(roots + dir_)
    for dir__ in dirs:
        # print(dir_ + "_" + dir__)
        for root_,dirs_,files in os.walk(roots + dir_ + "/" + dir__):
            # print(root_)
            imgs = []
            for file in files:
                srcImg = cv.imread(root_ + "/" + file,0)
                imgs.append(cut_img(srcImg,128,128))
            cells.append(imgs)
        labels.append(dir_ + "_" + dir__)
print(labels)
```

    ['hy', 'ml', 'zjg', 'fyc', 'zl', 'lsl', 'wq', 'zdx', 'rj', 'lqf', 'xch', 'yjf', 'zyf', 'syj', 'zc', 'wl', 'xxj', 'ljg', 'nhz', 'wyc']
    ['hy_00_2', 'hy_45_1', 'hy_45_3', 'hy_90_4', 'hy_00_4', 'hy_00_3', 'hy_90_1', 'hy_45_4', 'hy_00_1', 'hy_90_3', 'hy_90_2', 'hy_45_2', 'ml_00_2', 'ml_45_1', 'ml_45_3', 'ml_90_4', 'ml_00_4', 'ml_00_3', 'ml_90_1', 'ml_45_4', 'ml_00_1', 'ml_90_3', 'ml_90_2', 'ml_45_2', 'zjg_00_2', 'zjg_45_1', 'zjg_45_3', 'zjg_90_4', 'zjg_00_4', 'zjg_00_3', 'zjg_90_1', 'zjg_45_4', 'zjg_00_1', 'zjg_90_3', 'zjg_90_2', 'zjg_45_2', 'fyc_00_2', 'fyc_45_1', 'fyc_45_3', 'fyc_90_4', 'fyc_00_4', 'fyc_00_3', 'fyc_90_1', 'fyc_45_4', 'fyc_00_1', 'fyc_90_3', 'fyc_90_2', 'fyc_45_2', 'zl_00_2', 'zl_45_1', 'zl_45_3', 'zl_90_4', 'zl_00_4', 'zl_00_3', 'zl_90_1', 'zl_45_4', 'zl_00_1', 'zl_90_3', 'zl_90_2', 'zl_45_2', 'lsl_00_2', 'lsl_45_1', 'lsl_45_3', 'lsl_90_4', 'lsl_00_4', 'lsl_00_3', 'lsl_90_1', 'lsl_45_4', 'lsl_00_1', 'lsl_90_3', 'lsl_90_2', 'lsl_45_2', 'wq_00_2', 'wq_45_1', 'wq_45_3', 'wq_90_4', 'wq_00_4', 'wq_00_3', 'wq_90_1', 'wq_45_4', 'wq_00_1', 'wq_90_3', 'wq_90_2', 'wq_45_2', 'zdx_00_2', 'zdx_45_1', 'zdx_45_3', 'zdx_90_4', 'zdx_00_4', 'zdx_00_3', 'zdx_90_1', 'zdx_45_4', 'zdx_00_1', 'zdx_90_3', 'zdx_90_2', 'zdx_45_2', 'rj_00_2', 'rj_45_1', 'rj_45_3', 'rj_90_4', 'rj_00_4', 'rj_00_3', 'rj_90_1', 'rj_45_4', 'rj_00_1', 'rj_90_3', 'rj_90_2', 'rj_45_2', 'lqf_00_2', 'lqf_45_1', 'lqf_45_3', 'lqf_90_4', 'lqf_00_4', 'lqf_00_3', 'lqf_90_1', 'lqf_45_4', 'lqf_00_1', 'lqf_90_3', 'lqf_90_2', 'lqf_45_2', 'xch_00_2', 'xch_45_1', 'xch_45_3', 'xch_90_4', 'xch_00_4', 'xch_00_3', 'xch_90_1', 'xch_45_4', 'xch_00_1', 'xch_90_3', 'xch_90_2', 'xch_45_2', 'yjf_00_2', 'yjf_45_1', 'yjf_45_3', 'yjf_90_4', 'yjf_00_4', 'yjf_00_3', 'yjf_90_1', 'yjf_45_4', 'yjf_00_1', 'yjf_90_3', 'yjf_90_2', 'yjf_45_2', 'zyf_00_2', 'zyf_45_1', 'zyf_45_3', 'zyf_90_4', 'zyf_00_4', 'zyf_00_3', 'zyf_90_1', 'zyf_45_4', 'zyf_00_1', 'zyf_90_3', 'zyf_90_2', 'zyf_45_2', 'syj_00_2', 'syj_45_1', 'syj_45_3', 'syj_90_4', 'syj_00_4', 'syj_00_3', 'syj_90_1', 'syj_45_4', 'syj_00_1', 'syj_90_3', 'syj_90_2', 'syj_45_2', 'zc_00_2', 'zc_45_1', 'zc_45_3', 'zc_90_4', 'zc_00_4', 'zc_00_3', 'zc_90_1', 'zc_45_4', 'zc_00_1', 'zc_90_3', 'zc_90_2', 'zc_45_2', 'wl_00_2', 'wl_45_1', 'wl_45_3', 'wl_90_4', 'wl_00_4', 'wl_00_3', 'wl_90_1', 'wl_45_4', 'wl_00_1', 'wl_90_3', 'wl_90_2', 'wl_45_2', 'xxj_00_2', 'xxj_45_1', 'xxj_45_3', 'xxj_90_4', 'xxj_00_4', 'xxj_00_3', 'xxj_90_1', 'xxj_45_4', 'xxj_00_1', 'xxj_90_3', 'xxj_90_2', 'xxj_45_2', 'ljg_00_2', 'ljg_45_1', 'ljg_45_3', 'ljg_90_4', 'ljg_00_4', 'ljg_00_3', 'ljg_90_1', 'ljg_45_4', 'ljg_00_1', 'ljg_90_3', 'ljg_90_2', 'ljg_45_2', 'nhz_00_2', 'nhz_45_1', 'nhz_45_3', 'nhz_90_4', 'nhz_00_4', 'nhz_00_3', 'nhz_90_1', 'nhz_45_4', 'nhz_00_1', 'nhz_90_3', 'nhz_90_2', 'nhz_45_2', 'wyc_00_2', 'wyc_45_1', 'wyc_45_3', 'wyc_90_4', 'wyc_00_4', 'wyc_00_3', 'wyc_90_1', 'wyc_45_4', 'wyc_00_1', 'wyc_90_3', 'wyc_90_2', 'wyc_45_2']
    

# 将数据分为训练集与测试集两部分


```python
train_cells = []
test_cells = []
for cell in cells:
    train_cells.append(cell[:len(cell)//2])
    test_cells.append(cell[len(cell)//2:])
```

# 对训练集构建基于GEI的HOG特征
将训练集数据读入，首先提取GEI步态能量图

然后对GEI提取HOG特征，每张GEI的HOG特征为20736 * 1大小

标记240个训练数据，按照人进行编号，每个人12组数据，共20人

输出折线图横轴为HOG数据编号，共240个；纵轴为对应人员编号，共20个


```python
traingeis = []
for train_cell in train_cells:
    traingeis.append(get_GEI(train_cell))
    
trainhogs = []
for traingei in traingeis:
    trainhog = get_Hog(traingei)
    trainhogs.append(trainhog)
trainData = np.float32(trainhogs).reshape(-1,20736,1)
print(trainData.shape)
responses = np.repeat(np.arange(20),12)[:,np.newaxis]
print(responses.shape)
print(responses.T)
plt.plot(responses)
plt.show()
```

    (240, 20736, 1)
    (240, 1)
    [[ 0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
       2  2  2  2  2  2  2  2  2  2  2  2  3  3  3  3  3  3  3  3  3  3  3  3
       4  4  4  4  4  4  4  4  4  4  4  4  5  5  5  5  5  5  5  5  5  5  5  5
       6  6  6  6  6  6  6  6  6  6  6  6  7  7  7  7  7  7  7  7  7  7  7  7
       8  8  8  8  8  8  8  8  8  8  8  8  9  9  9  9  9  9  9  9  9  9  9  9
      10 10 10 10 10 10 10 10 10 10 10 10 11 11 11 11 11 11 11 11 11 11 11 11
      12 12 12 12 12 12 12 12 12 12 12 12 13 13 13 13 13 13 13 13 13 13 13 13
      14 14 14 14 14 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 15 15 15 15
      16 16 16 16 16 16 16 16 16 16 16 16 17 17 17 17 17 17 17 17 17 17 17 17
      18 18 18 18 18 18 18 18 18 18 18 18 19 19 19 19 19 19 19 19 19 19 19 19]]
    


![png](/picture/output_14_1.png)


# 对测试集构建基于GEI的HOG特征

与训练集处理方式相同，每个HOG特征为20736 * 1大小


```python
testgeis = []
for test_cell in test_cells:
    testgeis.append(get_GEI(test_cell))
testhogs = []
for testgei in testgeis:
    testhog = get_Hog(testgei)
    testhogs.append(testhog)
testData = np.float32(testhogs).reshape(-1,20736,1)
print(testData.shape)
```

    (240, 20736, 1)
    

# 采用线性内核函数训练OpenCV内置SVM分类器


```python
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(5.35)
svm.setGamma(3.58)
svm.train(trainData,cv.ml.ROW_SAMPLE,responses)
svm.save('svm_data.dat')
```

# 使用训练好的模型进行预测
打印预测结果

采用折线图输出预测结果，横纵坐标同训练集标记类型

实验结果识别率达到100%

```python
result = svm.predict(testData)[1].astype("int")
print(result.T)
plt.plot(result)
plt.show()
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)
```

    [[ 0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
       2  2  2  2  2  2  2  2  2  2  2  2  3  3  3  3  3  3  3  3  3  3  3  3
       4  4  4  4  4  4  4  4  4  4  4  4  5  5  5  5  5  5  5  5  5  5  5  5
       6  6  6  6  6  6  6  6  6  6  6  6  7  7  7  7  7  7  7  7  7  7  7  7
       8  8  8  8  8  8  8  8  8  8  8  8  9  9  9  9  9  9  9  9  9  9  9  9
      10 10 10 10 10 10 10 10 10 10 10 10 11 11 11 11 11 11 11 11 11 11 11 11
      12 12 12 12 12 12 12 12 12 12 12 12 13 13 13 13 13 13 13 13 13 13 13 13
      14 14 14 14 14 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 15 15 15 15
      16 16 16 16 16 16 16 16 16 16 16 16 17 17 17 17 17 17 17 17 17 17 17 17
      18 18 18 18 18 18 18 18 18 18 18 18 19 19 19 19 19 19 19 19 19 19 19 19]]
    


![png](/picture/output_20_1.png)


    100.0
    
 # 将数据分为训练集与测试集两部分
训练集每人每方向序列帧图像的一半

测试集为17组，每组每人每方向序列帧图像数目相同，为组号（1，2...17）


```python
minlength = len(cells[0])//2
for cell in cells:
    if minlength > len(cell)//2:
        minlength = len(cell)//2
    else:
        continue
        
        
train_cells = []
for cell in cells:
    train_cells.append(cell[:len(cell)//2])
    
test = []
for i in range(1,minlength):
    test_cells = []
    for cell in cells:
        test_cells.append(cell[-minlength+i:])
    test.append(test_cells)
print(len(train_cells))
print(len(test))
```

    240
    17
    
    
# 对训练集构建基于GEI的HOG特征
将训练集数据读入，首先提取GEI步态能量图

然后对GEI提取HOG特征，每张GEI的HOG特征为20736 * 1大小

标记240个训练数据，按照人进行编号，每个人12组数据，共20人

输出折线图横轴为HOG数据编号，共240个；纵轴为对应人员编号，共20个


```python
responses = []

traingeis = []
for train_cell in train_cells:
    traingeis.append(get_GEI(train_cell))

trainhogs = []
for traingei in traingeis:
    trainhog = get_Hog(traingei)
    trainhogs.append(trainhog)
trainData = np.float32(trainhogs).reshape(-1,20736,1)
responses = np.repeat(np.arange(20),12)[:,np.newaxis]
print(trainData.shape)
```

    (240, 20736, 1)
    

# 对测试集构建基于GEI的HOG特征

与训练集处理方式相同，每个HOG特征为20736 * 1大小


```python
def maketestData(test_cells):
    testgeis = []
    for test_cell in test_cells:
        testgeis.append(get_GEI(test_cell))
    testhogs = []
    for testgei in testgeis:
        testhog = get_Hog(testgei)
        testhogs.append(testhog)
    testData = np.float32(testhogs).reshape(-1,20736,1)
    return testData

testDatas = []
for test_cells in test:
    testDatas.append(maketestData(test_cells))
print(len(testDatas))
```

    17
    

# 采用线性内核函数训练OpenCV内置SVM分类器


```python
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(8.35)
svm.setGamma(3.58)
svm.train(trainData,cv.ml.ROW_SAMPLE,responses)
svm.save('svm_data.dat')
```

# 使用训练好的模型进行预测
分析测试集大小对模型预测准确度的影响


```python
results = []
corrects = []
def svmpredict(i):
    result = svm.predict(testDatas[len(testDatas)-i-1])[1].astype("int")
    mask = result==responses
    correct = np.count_nonzero(mask)
    results.append(result)
    corrects.append(correct*100.0/result.size)
    print(correct*100.0/result.size)

for i in range(len(test)):
    svmpredict(i)
plt.title("Predict correct rate")
plt.xlabel("Number of images choosen to test model")
plt.ylabel("Correct rate")
plt.plot(corrects)
plt.show()
```

    17.916666666666668
    25.416666666666668
    35.833333333333336
    49.166666666666664
    57.5
    66.66666666666667
    70.41666666666667
    74.16666666666667
    79.16666666666667
    84.58333333333333
    88.33333333333333
    90.41666666666667
    91.25
    92.5
    93.75
    96.25
    95.83333333333333
    


![png](output_18_1.png)

    

# 输出测试集与训练集对用HOG,GEI直观对比


```python
for i in range(len(trainhogs)):
    plt.subplot(141), plt.title("训练集HOG特征"),plt.hist(trainhogs[i])
    plt.subplot(142), plt.title("训练集GEI图"),plt.imshow(traingeis[i],cmap='gray')
    plt.subplot(143), plt.title("测试集HOG特征"),plt.hist(testhogs[i])
    plt.subplot(144), plt.title("测试集GEI图"),plt.imshow(testgeis[i],cmap='gray')
    plt.show()
```


![png](/picture/output_22_0.png)



![png](/picture/output_22_1.png)



![png](/picture/output_22_2.png)



![png](/picture/output_22_3.png)



![png](/picture/output_22_4.png)



![png](/picture/output_22_5.png)



![png](/picture/output_22_6.png)



![png](/picture/output_22_7.png)



![png](/picture/output_22_8.png)



![png](/picture/output_22_9.png)



![png](/picture/output_22_10.png)



![png](/picture/output_22_11.png)



![png](/picture/output_22_12.png)



![png](/picture/output_22_13.png)



![png](/picture/output_22_14.png)



![png](/picture/output_22_15.png)



![png](/picture/output_22_16.png)



![png](/picture/output_22_17.png)



![png](/picture/output_22_18.png)



![png](/picture/output_22_19.png)



![png](/picture/output_22_20.png)



![png](/picture/output_22_21.png)



![png](/picture/output_22_22.png)



![png](/picture/output_22_23.png)



![png](/picture/output_22_24.png)



![png](/picture/output_22_25.png)



![png](/picture/output_22_26.png)



![png](/picture/output_22_27.png)



![png](/picture/output_22_28.png)



![png](/picture/output_22_29.png)



![png](/picture/output_22_30.png)



![png](/picture/output_22_31.png)



![png](/picture/output_22_32.png)



![png](/picture/output_22_33.png)



![png](/picture/output_22_34.png)



![png](/picture/output_22_35.png)



![png](/picture/output_22_36.png)



![png](/picture/output_22_37.png)



![png](/picture/output_22_38.png)



![png](/picture/output_22_39.png)



![png](/picture/output_22_40.png)



![png](/picture/output_22_41.png)



![png](/picture/output_22_42.png)



![png](/picture/output_22_43.png)



![png](/picture/output_22_44.png)



![png](/picture/output_22_45.png)



![png](/picture/output_22_46.png)



![png](/picture/output_22_47.png)



![png](/picture/output_22_48.png)



![png](/picture/output_22_49.png)



![png](/picture/output_22_50.png)



![png](/picture/output_22_51.png)



![png](/picture/output_22_52.png)



![png](/picture/output_22_53.png)



![png](/picture/output_22_54.png)



![png](/picture/output_22_55.png)



![png](/picture/output_22_56.png)



![png](/picture/output_22_57.png)



![png](/picture/output_22_58.png)



![png](/picture/output_22_59.png)



![png](/picture/output_22_60.png)



![png](/picture/output_22_61.png)



![png](/picture/output_22_62.png)



![png](/picture/output_22_63.png)



![png](/picture/output_22_64.png)



![png](/picture/output_22_65.png)



![png](/picture/output_22_66.png)



![png](/picture/output_22_67.png)



![png](/picture/output_22_68.png)



![png](/picture/output_22_69.png)



![png](/picture/output_22_70.png)



![png](/picture/output_22_71.png)



![png](/picture/output_22_72.png)



![png](/picture/output_22_73.png)



![png](/picture/output_22_74.png)



![png](/picture/output_22_75.png)



![png](/picture/output_22_76.png)



![png](/picture/output_22_77.png)



![png](/picture/output_22_78.png)



![png](/picture/output_22_79.png)



![png](/picture/output_22_80.png)



![png](/picture/output_22_81.png)



![png](/picture/output_22_82.png)



![png](/picture/output_22_83.png)



![png](/picture/output_22_84.png)



![png](/picture/output_22_85.png)



![png](/picture/output_22_86.png)



![png](/picture/output_22_87.png)



![png](/picture/output_22_88.png)



![png](/picture/output_22_89.png)



![png](/picture/output_22_90.png)



![png](/picture/output_22_91.png)



![png](/picture/output_22_92.png)



![png](/picture/output_22_93.png)



![png](/picture/output_22_94.png)



![png](/picture/output_22_95.png)



![png](/picture/output_22_96.png)



![png](/picture/output_22_97.png)



![png](/picture/output_22_98.png)



![png](/picture/output_22_99.png)



![png](/picture/output_22_100.png)



![png](/picture/output_22_101.png)



![png](/picture/output_22_102.png)



![png](/picture/output_22_103.png)



![png](/picture/output_22_104.png)



![png](/picture/output_22_105.png)



![png](/picture/output_22_106.png)



![png](/picture/output_22_107.png)



![png](/picture/output_22_108.png)



![png](/picture/output_22_109.png)



![png](/picture/output_22_110.png)



![png](/picture/output_22_111.png)



![png](/picture/output_22_112.png)



![png](/picture/output_22_113.png)



![png](/picture/output_22_114.png)



![png](/picture/output_22_115.png)



![png](/picture/output_22_116.png)



![png](/picture/output_22_117.png)



![png](/picture/output_22_118.png)



![png](/picture/output_22_119.png)



![png](/picture/output_22_120.png)



![png](/picture/output_22_121.png)



![png](/picture/output_22_122.png)



![png](/picture/output_22_123.png)



![png](/picture/output_22_124.png)



![png](/picture/output_22_125.png)



![png](/picture/output_22_126.png)



![png](/picture/output_22_127.png)



![png](/picture/output_22_128.png)



![png](/picture/output_22_129.png)



![png](/picture/output_22_130.png)



![png](/picture/output_22_131.png)



![png](/picture/output_22_132.png)



![png](/picture/output_22_133.png)



![png](/picture/output_22_134.png)



![png](/picture/output_22_135.png)



![png](/picture/output_22_136.png)



![png](/picture/output_22_137.png)



![png](/picture/output_22_138.png)



![png](/picture/output_22_139.png)



![png](/picture/output_22_140.png)



![png](/picture/output_22_141.png)



![png](/picture/output_22_142.png)



![png](/picture/output_22_143.png)



![png](/picture/output_22_144.png)



![png](/picture/output_22_145.png)



![png](/picture/output_22_146.png)



![png](/picture/output_22_147.png)



![png](/picture/output_22_148.png)



![png](/picture/output_22_149.png)



![png](/picture/output_22_150.png)



![png](/picture/output_22_151.png)



![png](/picture/output_22_152.png)



![png](/picture/output_22_153.png)



![png](/picture/output_22_154.png)



![png](/picture/output_22_155.png)



![png](/picture/output_22_156.png)



![png](/picture/output_22_157.png)



![png](/picture/output_22_158.png)



![png](/picture/output_22_159.png)



![png](/picture/output_22_160.png)



![png](/picture/output_22_161.png)



![png](/picture/output_22_162.png)



![png](/picture/output_22_163.png)



![png](/picture/output_22_164.png)



![png](/picture/output_22_165.png)



![png](/picture/output_22_166.png)



![png](/picture/output_22_167.png)



![png](/picture/output_22_168.png)



![png](/picture/output_22_169.png)



![png](/picture/output_22_170.png)



![png](/picture/output_22_171.png)



![png](/picture/output_22_172.png)



![png](/picture/output_22_173.png)



![png](/picture/output_22_174.png)



![png](/picture/output_22_175.png)



![png](/picture/output_22_176.png)



![png](/picture/output_22_177.png)



![png](/picture/output_22_178.png)



![png](/picture/output_22_179.png)



![png](/picture/output_22_180.png)



![png](/picture/output_22_181.png)



![png](/picture/output_22_182.png)



![png](/picture/output_22_183.png)



![png](/picture/output_22_184.png)



![png](/picture/output_22_185.png)



![png](/picture/output_22_186.png)



![png](/picture/output_22_187.png)



![png](/picture/output_22_188.png)



![png](/picture/output_22_189.png)



![png](/picture/output_22_190.png)



![png](/picture/output_22_191.png)



![png](/picture/output_22_192.png)



![png](/picture/output_22_193.png)



![png](/picture/output_22_194.png)



![png](/picture/output_22_195.png)



![png](/picture/output_22_196.png)



![png](/picture/output_22_197.png)



![png](/picture/output_22_198.png)



![png](/picture/output_22_199.png)



![png](/picture/output_22_200.png)



![png](/picture/output_22_201.png)



![png](/picture/output_22_202.png)



![png](/picture/output_22_203.png)



![png](/picture/output_22_204.png)



![png](/picture/output_22_205.png)



![png](/picture/output_22_206.png)



![png](/picture/output_22_207.png)



![png](/picture/output_22_208.png)



![png](/picture/output_22_209.png)



![png](/picture/output_22_210.png)



![png](/picture/output_22_211.png)



![png](/picture/output_22_212.png)



![png](/picture/output_22_213.png)



![png](/picture/output_22_214.png)



![png](/picture/output_22_215.png)



![png](/picture/output_22_216.png)



![png](/picture/output_22_217.png)



![png](/picture/output_22_218.png)



![png](/picture/output_22_219.png)



![png](/picture/output_22_220.png)



![png](/picture/output_22_221.png)



![png](/picture/output_22_222.png)



![png](/picture/output_22_223.png)



![png](/picture/output_22_224.png)



![png](/picture/output_22_225.png)



![png](/picture/output_22_226.png)



![png](/picture/output_22_227.png)



![png](/picture/output_22_228.png)



![png](/picture/output_22_229.png)



![png](/picture/output_22_230.png)



![png](/picture/output_22_231.png)



![png](/picture/output_22_232.png)



![png](/picture/output_22_233.png)



![png](/picture/output_22_234.png)



![png](/picture/output_22_235.png)



![png](/picture/output_22_236.png)



![png](/picture/output_22_237.png)



![png](/picture/output_22_238.png)



![png](/picture/output_22_239.png)

