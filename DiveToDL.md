# 3. 线性网络

## 3.1 softmax

**模型**

![image-20210301200444557](https://gitee.com/sxy22/note_images/raw/master/image-20210301200444557.png)

![image-20210301200456854](https://gitee.com/sxy22/note_images/raw/master/image-20210301200456854.png)

## 3.2 交叉熵损失



![image-20210301200637163](https://gitee.com/sxy22/note_images/raw/master/image-20210301200637163.png)

# 4 多层感知机

## 多层网络

![image-20210301200743658](https://gitee.com/sxy22/note_images/raw/master/image-20210301200743658.png)

## 激活函数

![image-20210301200808276](https://gitee.com/sxy22/note_images/raw/master/image-20210301200808276.png)

### RELU

![image-20211216150204181](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216150204181.png)

![image-20211216150233719](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216150233719.png)



![image-20211216150239775](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216150239775.png)



使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现的更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题



### sigmoid

![image-20211216150310847](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216150310847.png)

![image-20211216150320703](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216150320703.png)



# 权重衰减

![image-20211216151224473](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216151224473.png)



# Dropout

我们希望模型深度挖掘特征，即将其权重分散到许多特征中， 而不是过于依赖少数潜在的虚假关联。

经典泛化理论认为，为了缩小训练和测试性能之间的差距，应该以简单的模型为目标。

![image-20211216152121297](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216152121297.png)



# 数值稳定性和模型初始化

## 梯度消失和梯度爆炸

![image-20211216153415572](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216153415572.png)



![image-20211216153422201](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211216153422201.png)











# 归一化层 barch normalization

训练深层神经网络是十分困难的，特别是在较短的时间内使他们收敛更加棘手。 在本节中，我们将介绍*批量规范化*（batch normalization这是一种流行且有效的技术，可持续加速深层网络的收敛速度。

![image-20211218152926709](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211218152926709.png)

## 全连接层 归一

![image-20211218153218483](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211218153218483.png)

## 卷积层 归一

![image-20211218153308531](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211218153308531.png)



### 预测过程中的批量规范化

![image-20211218154643355](https://raw.githubusercontent.com/sxy22/notes_pic/main/image-20211218154643355.png)



# 6 卷积神经网络

### 二维卷积层

![image-20210301205355409](https://gitee.com/sxy22/note_images/raw/master/image-20210301205355409.png)

### 填充和步幅

![image-20210301210046996](https://gitee.com/sxy22/note_images/raw/master/image-20210301210046996.png)

#### 填充

![image-20210301210222177](https://gitee.com/sxy22/note_images/raw/master/image-20210301210222177.png)

![image-20210301210254331](https://gitee.com/sxy22/note_images/raw/master/image-20210301210254331.png)

#### 步幅

![image-20210301210719308](https://gitee.com/sxy22/note_images/raw/master/image-20210301210719308.png)

### 多输⼊通道和多输出通道

![image-20210409211503129](https://gitee.com/sxy22/note_images/raw/master/image-20210409211503129.png)

![image-20210409211547543](https://gitee.com/sxy22/note_images/raw/master/image-20210409211547543.png)



![image-20210409212150215](https://gitee.com/sxy22/note_images/raw/master/image-20210409212150215.png)





### 池化层

#### ⼆维最⼤池化层和平均池化层

![image-20210301213422259](https://gitee.com/sxy22/note_images/raw/master/image-20210301213422259.png)

#### 填充和步幅



### LeNet

![image-20210409220752228](https://gitee.com/sxy22/note_images/raw/master/image-20210409220752228.png)

卷积层块⾥的基本单位是卷积层后接最⼤池化层：

卷积层块由两个这样的基本单位重复堆叠构成。

在卷积层块中，每个卷积层都使⽤5 x 5的窗口，并在输出上使⽤sigmoid激活函数。

第⼀个卷积层输出通道数为6，第⼆个卷积层输出通道数则增加到16。

卷积层块的两个最⼤池化层的窗口形状均为2  2，且步幅为2。由于池化窗口与步幅形状相同，池化窗口在输⼊上每次滑动所覆盖的区域互不重叠。

卷积层块的输出形状为(批量⼤小, 通道, ⾼, 宽)。当卷积层块的输出传⼊全连接层块时，全连接
层块会将小批量中每个样本变平（flatten）。也就是说，全连接层的输⼊形状将变成⼆维，其中第
⼀维是小批量中的样本，第⼆维是每个样本变平后的向量表⽰，且向量⻓度为通道、⾼和宽的乘
积。全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数







### VGG

![image-20210302202900935](https://gitee.com/sxy22/note_images/raw/master/image-20210302202900935.png)

![image-20210302204600079](https://gitee.com/sxy22/note_images/raw/master/image-20210302204600079.png)

### NiN

![image-20210302204916873](https://gitee.com/sxy22/note_images/raw/master/image-20210302204916873.png)

![image-20210411220355116](https://gitee.com/sxy22/note_images/raw/master/image-20210411220355116.png)





### GoogLeNet

![image-20210302212538365](https://gitee.com/sxy22/note_images/raw/master/image-20210302212538365.png)

![image-20210302212939272](https://gitee.com/sxy22/note_images/raw/master/image-20210302212939272.png)

![image-20210302213057746](https://gitee.com/sxy22/note_images/raw/master/image-20210302213057746.png)

![image-20210302213137407](https://gitee.com/sxy22/note_images/raw/master/image-20210302213137407.png)

![image-20210302213425306](https://gitee.com/sxy22/note_images/raw/master/image-20210302213425306.png)

![](https://gitee.com/sxy22/note_images/raw/master/image-20210302213425306.png)

![image-20210302213600989](C:\Users\91317\AppData\Roaming\Typora\typora-user-images\image-20210302213600989.png)



## 残差网络 ResNet

![image-20210415210447960](https://gitee.com/sxy22/note_images/raw/master/image-20210415210447960.png)

![image-20210415210636695](https://gitee.com/sxy22/note_images/raw/master/image-20210415210636695.png)

![image-20210415212222903](https://gitee.com/sxy22/note_images/raw/master/image-20210415212222903.png)



### 稠密连接网络

![image-20210415220614756](https://gitee.com/sxy22/note_images/raw/master/image-20210415220614756.png)



# 循环神经网络

### 语言模型 

![image-20210416205726845](https://gitee.com/sxy22/note_images/raw/master/image-20210416205726845.png)

### 含隐藏状态的循环神经网络

![image-20210416210710577](https://gitee.com/sxy22/note_images/raw/master/image-20210416210710577.png)

![image-20210416211049143](https://gitee.com/sxy22/note_images/raw/master/image-20210416211049143.png)

### 采样

#### 随机采样

![image-20210416214731790](https://gitee.com/sxy22/note_images/raw/master/image-20210416214731790.png)



### 反向传播 梯度不稳定

![image-20210417215028422](https://gitee.com/sxy22/note_images/raw/master/image-20210417215028422.png)

![image-20210417215039956](https://gitee.com/sxy22/note_images/raw/master/image-20210417215039956.png)

### 门控

![image-20210419223521156](https://gitee.com/sxy22/note_images/raw/master/image-20210419223521156.png)

![image-20210419223529294](https://gitee.com/sxy22/note_images/raw/master/image-20210419223529294.png)

![image-20210419223715690](https://gitee.com/sxy22/note_images/raw/master/image-20210419223715690.png)

![image-20210419223720676](https://gitee.com/sxy22/note_images/raw/master/image-20210419223720676.png)

![image-20210419224136548](https://gitee.com/sxy22/note_images/raw/master/image-20210419224136548.png)

![image-20210419223922321](https://gitee.com/sxy22/note_images/raw/master/image-20210419223922321.png)

### LSTM

![image-20210420200828398](https://gitee.com/sxy22/note_images/raw/master/image-20210420200828398.png)

![image-20210420200839462](https://gitee.com/sxy22/note_images/raw/master/image-20210420200839462.png)

![image-20210420200847209](https://gitee.com/sxy22/note_images/raw/master/image-20210420200847209.png)

![image-20210420200857750](https://gitee.com/sxy22/note_images/raw/master/image-20210420200857750.png)

![image-20210420200907840](https://gitee.com/sxy22/note_images/raw/master/image-20210420200907840.png)

### 深度CNN

![image-20210420201806381](https://gitee.com/sxy22/note_images/raw/master/image-20210420201806381.png)

![image-20210420202458413](https://gitee.com/sxy22/note_images/raw/master/image-20210420202458413.png)

### 双向循环神经⽹络

![image-20210420211836385](https://gitee.com/sxy22/note_images/raw/master/image-20210420211836385.png)

![image-20210420212143943](https://gitee.com/sxy22/note_images/raw/master/image-20210420212143943.png)



# 优化算法

### 动量法

![image-20210421232012562](https://gitee.com/sxy22/note_images/raw/master/image-20210421232012562.png)

![image-20210421232029947](https://gitee.com/sxy22/note_images/raw/master/image-20210421232029947.png)

![image-20210421232041809](https://gitee.com/sxy22/note_images/raw/master/image-20210421232041809.png)

### AdaGrad算法

![image-20210421235612565](https://gitee.com/sxy22/note_images/raw/master/image-20210421235612565.png)

![image-20210421235709499](https://gitee.com/sxy22/note_images/raw/master/image-20210421235709499.png)

### RMSProp算法

![image-20210421235952806](https://gitee.com/sxy22/note_images/raw/master/image-20210421235952806.png)

### AdaDelta算法

![image-20210422000332763](https://gitee.com/sxy22/note_images/raw/master/image-20210422000332763.png)

![image-20210422000338275](https://gitee.com/sxy22/note_images/raw/master/image-20210422000338275.png)

### Adam算法

Adam算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均

![image-20210422000651012](https://gitee.com/sxy22/note_images/raw/master/image-20210422000651012.png)

![image-20210422000657585](https://gitee.com/sxy22/note_images/raw/master/image-20210422000657585.png)

## NLP

### 词嵌⼊（word2vec）

![image-20210426223039241](https://gitee.com/sxy22/note_images/raw/master/image-20210426223039241.png)

#### 跳字模型

![image-20210426223142411](https://gitee.com/sxy22/note_images/raw/master/image-20210426223142411.png)

![image-20210426223513963](https://gitee.com/sxy22/note_images/raw/master/image-20210426223513963.png)

![image-20210426224142681](https://gitee.com/sxy22/note_images/raw/master/image-20210426224142681.png)

#### 连续词袋模型

![image-20210426224245285](https://gitee.com/sxy22/note_images/raw/master/image-20210426224245285.png)

![image-20210426224304943](https://gitee.com/sxy22/note_images/raw/master/image-20210426224304943.png)

![image-20210426224636322](https://gitee.com/sxy22/note_images/raw/master/image-20210426224636322.png)

### 近似训练

![image-20210427201702068](https://gitee.com/sxy22/note_images/raw/master/image-20210427201702068.png)

#### 负采样

![image-20210427201958252](https://gitee.com/sxy22/note_images/raw/master/image-20210427201958252.png)

![image-20210427202655205](https://gitee.com/sxy22/note_images/raw/master/image-20210427202655205.png)

![image-20210427202702383](https://gitee.com/sxy22/note_images/raw/master/image-20210427202702383.png)

#### ⼆次采样

![](https://gitee.com/sxy22/note_images/raw/master/image-20210427215224974.png)