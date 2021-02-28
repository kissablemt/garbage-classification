# garbage-classification
基于TensorFlow和Keras的垃圾分类系统

1、data_gen.py用于数据生成
2、models.py曾使用过的模型
3、train.py用于模型的训练
4、predict.py用于模型可视化预测（tf>=2.0)
5、kaggle_submit.py最终kaggle上的提交源码。
6、history为历史模型的History(loss,acc)

## **背景**
如今，垃圾分类已成为社会热点话题。其实在2019年4月26日，我国住房和城乡建设部等部门就发布了《关于在全国地级及以上城市全面开展生活垃圾分类工作的通知》，决定自2019年起在全国地级及以上城市全面启动生活垃圾分类工作。到2020年底，46个重点城市基本建成生活垃圾分类处理系统。[1]

基于深度学习设计一款垃圾分类识别程序。通过对垃圾进行拍照对垃圾图片进行快速识别分类，通过查询垃圾分类规则，输出该垃圾图片中物品属于的种类。

## **数据来源**
![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.001.png)

图  SEQ 图 \\* ARABIC 1 <https://www.kaggle.com/asdasdasasdas/garbage-classification>
## **数据读取与增强**
1、使用keras的ImageDataGenerator，用以生成一个batch的图像数据，可实时数据增强。

2、应分为训练集（train）与验证集（validation），其中训练集需要进行数据增强。

3、ImageDataGenerator.flow\_from\_directory(directory)进行文件流的读取，从中产生batch数据。

4、图像归一化、随机旋转、随机水平翻转、随机垂直翻转、随机投影变换等。

## **原理**
在人脑视觉机理：从原始信号， 做低级抽象，逐渐向高级抽象迭代下，提出深度学习。

深度学习可通过学习一种深层非线性网络结构，实现复杂函数逼近，表征输入数据分布式表示，并展现了强大的从少数样本集中学习数据集本质特征的能力。

![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.002.png)
## **模型建立**
![model](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.003.png "model")

图  SEQ 图 \\* ARABIC 2 model2网络架构图

1、进行Convolution2D，进行局部特征提取；

2、Pooling降低图片的空间尺寸，增大卷积核感受野，提取更抽象（高层）特征，预防过拟合；

3、重复1、2三次，其中使用不同kernel；

4、Flatten-》Dense。“压平”，然后过渡到全连接层。softmax更适合垃圾分类这种多分类。

5、Dropout。模型的参数比较多，尤其是这次垃圾分类的数据样本太少，容易发生过拟合现象，需要适当断开神经网络的连接，减少神经元之间复杂的共适应关系，使得神经元连接的鲁棒性提高；


## **问题分析**
在多次调整网络架构，使用了不同optimizers时，模型的正确率不能超过80%时，初步分析：

- 扩大训练的数据集和数据增强
- 进行迁移学习

由于在Kaggle上进行测试，数据集固定，不考虑扩大数据集，只能进行数据增强与迁移学习。

在多次测试下，观察其中Model-Loss图，train\_loss一直下降，收敛，而valid\_loss周期性变动，不收敛，发生震荡，是过拟合的现象，

![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.004.png)

图  SEQ 图 \\* ARABIC 3 验证曲线分析

分析：

- 数据集过小，只有2000张图
- batch\_size设置过小
- 没有设置合理Dropout的范围
- epochs设置过小

## **模型改进**
从Kaggle一篇文章CNN Architectures[2]中的图  SEQ 图 \\* ARABIC 4 ，可看到InceptionV3和ResNet50在多分类识别问题上效果比较好，故分别以InceptionV3、ResNet50作为base\_model，并载入预训练集，对训练集进行训练。

![IMG\_256](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.005.png "IMG\_256")

图  SEQ 图 \\* ARABIC 5 多种模型的正确率（来源[2]）

![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.006.png)

图  SEQ 图 \\* ARABIC 6 InceptionV3-多分类识别（来源[2]）


参考仵赛飞[3]在github上关于「华为云人工智能大赛·垃圾分类挑战杯」比赛的代码，采用了他实验中正确率最高的模型，即使用EfficientNetB5作为base\_model，并使用自定义的learning\_rate：SGDR余弦退火学习率，进行训练，在华为比赛中最终正确率可达95%左右。

`    `将分别以InceptionV3、ResNet50、EfficientNetB5作为模型的的预训练base\_model下，进行参数调整，已获得最好的训练结果。


|模型名称|主要内容|epochs|batch\_size|validation\_score|
| :-: | :-: | :-: | :-: | :-: |
|model1|<p>（二维卷积、池化层）\*3</p><p>全连接\*2</p>|100|32|0.78|
|model2|<p>model2基础上增加一个全连接层</p><p>在每一个全连接层后Dropout</p>|100|32|0.80|
|model3|finetune-InceptionV3-pretrained|100|32|0.80|
|model4|<p>在model3基础上</p><p>增加Dense的输出维度（512-》1024）</p><p>优化器（adam-》nadam）</p>|100|32|0.84|
|model5|<p>在model4基础上</p><p>减少Dropout的断开率（0.05-》0.02）</p><p>Dense（6-》2）</p>|500|32|0.88|
|model4|使用model4|500|100|0.92|
|model4|使用model4|500|200|0.97|
|model6|<p>在model4基础上</p><p>增加Dropout（0.05-》0.25）</p>|500|200|0.94|
|model7|finetune-ResNet50-pretrained|100|200|0.44|
|model8|<p>在model7基础上</p><p>增加Dropout（0-》0.5）</p>|100|200|0.41|
|model9|<p>finetune-EffNetB5-pretrained</p><p>SGDR余弦退火学习率调整</p>|16|100|0.57|
表 1 主要模型改进

## **实验结果**
![下载](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.007.png "下载")

图  SEQ 图 \\* ARABIC 7 valid组抽取16张图片作为可视化验证

![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.008.png)

图  SEQ 图 \\* ARABIC 8 Log

![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.009.png)

图  SEQ 图 \\* ARABIC 9 无迁移学习下最优模型model2-accuracy

![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.010.png)

图  SEQ 图 \\* ARABIC 10 无迁移学习下最优模型model2-loss

![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.011.png)

图  SEQ 图 \\* ARABIC 11 迁移学习下最终模型model4-accuracy

![](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.012.png)

图  SEQ 图 \\* ARABIC 12 迁移学习下最终模型model4-loss

仵赛飞[3]的EfficientNetB5加SGDR余弦退火学习率调整效果并不好，可能是因为对比华为比赛，数据集实在太小。另外仵赛飞的图片增强做得更好，增加了随机crop。

在github上看到华为比赛中有许多人使用ResNet50作迁移学习，加上Shivam Bansal[2]写的文章，也在实验中使用了ResNet50，但效果也不好，甚至没有超过无迁移学习的模型。

![model](img/Aspose.Words.b03dc22b-67c3-43c9-abdc-30fbb826c6a1.013.png "model")

图  SEQ 图 \\* ARABIC 13 最优模型（Nadam优化器）

最终使用预训练的InceptionV3，在使用Nadam优化器下获得最高正确率97%，

在只有这么少数据的情况下达到97%正确率已经不错，可使用InceptionV3做迁移学习进行华为垃圾分类数据集的测试。

## **总结**
最开始是在github上找有关垃圾分类的源码，发现很多都出自「华为云人工智能大赛·垃圾分类挑战杯」。代码繁杂，而且数据集大，加上本地电脑无法gpu加速，操作起来很困难。为此看了许多文章，碰巧遇上一博主用的Kaggle上的数据，数据量少，而且Kaggle上有gpu加速，很适合学习。随后开通4个kaggle账号，提交不同的模型以获得最优成绩。在慢慢理解验证集曲线后，共跑了17个Version，找到了在InceptionV3迁移学习下的最优模型。

接下来讲继续使用该模型去训练华为比赛的训练集，测试模型的效果。日后有机会在kaggle上参加一些比赛，提升自我。


## **参考**
1. 「华为云人工智能大赛·垃圾分类挑战杯」
1. Shivam Bansal.CNN Architectures.

[https://www.kaggle.com/shivamb/cnn-architectures-vgg-resnet-inception-tl]([2]https:/www.kaggle.com/shivamb/cnn-architectures-vgg-resnet-inception-tl)

1. 仵赛飞. <https://github.com/wusaifei/garbage_classify>
1. keras中文文档. <https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/>
