# Adversarial Camouflage Utilizing style Transfer

## 介绍

此次实验是对[Adversarial Camouflage: Hiding Physical-World Attacks with Natural Styles](https://arxiv.org/abs/2003.08757)思想的实现，并且只针对**数字攻击**。实验参考了[原作代码](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles)。我们使用的模型与论文中一致，均为`vgg19`，在`ImageNet`上预训练好的模型。本实验使用pytorch实现。

## 实验初步

### Requirements

此时实验未使用GPU，在CPU上运行得到结果

> numpy==1.23.5
>
> Pillow==9.4.0
>
> torch==2.4.0
>
> torchvision==0.19.0

### Directory

```plaintext
|-- data
|   |-- content
|   |   |-- stop-sign
|   |-- style
|   |   |-- stop-sign
|   |-- content-mask
|   |-- style-mask
|
|-- Vgg
|   |-- Vgg19.py
|
|-- main.py
|-- attack.py
|-- loss.py
|-- utils.py
|-- Result
|-- synset.txt
```
其中`main`为整个程序的入口，其中可以进行各种超参数、参数的修改，图像路径的选择；`attack.py`中实现攻击函数，主要进行基于风格迁移的对抗性样本生成；攻击过程中使用的模型在`Vgg`中py文件定义，在Vgg19类中，在forward过程中，对需要利用到的中间层特征以及相应的logits向量等进行了维护并返回；

`loss.py`中实现了各种攻击优化过程中需要的损失函数；`utils.py`中包含了攻击过程中需用到的各种功能函数，如进行图像加载、图像保存、风格掩码生成、最大概率类别获取、gram矩阵生成等；`synset.txt`中有供生成真实label的1000个ImageNet类别。

`data`中有对抗使用的原始图像、风格图像、原始分割图像、风格分割图像，分割图像主要用于掩码生成，以针对特定的攻击区域；最终生成的对抗性样本将保存到`result`中。

### Run Attack

直接运行main.py，使用不同的图像路径，以及相应各种的参数在main中修改

> main.py

## 方法

### 对抗性:

实现了两种形式的对抗，一种有明确目标标签，另一无目标标签。

![image](https://github.com/user-attachments/assets/d0df844a-0eb5-4d97-aac0-6013bc70392b)




### 风格迁移:

为了使得对抗性样本与风格图像相似，通过对网络中的对应层特征对比的方法设计损失函数。G 是从特征提取器 F 中在第 l 层提取的特征的 Gram 矩阵。实验中使用了5层不同的特征，分别是conv1_1,conv2_1,conv3_1,conv4_1,conv5_1,详见实验代码。

![image](https://github.com/user-attachments/assets/7e3ef4b0-a33d-418d-928f-40899030d79c)



### 内容一致:

为了保持对抗性干扰隐蔽性，对抗性样本不能过度偏离原始图像，这里仍然使用某一网络层得到的特征进行一致性度量，实验中使用了conv4_2得到的特征。

![image](https://github.com/user-attachments/assets/fc51fb19-4a53-4b4d-92ad-b6f7da3c2d8c)



### 损失函数:

最终的损失函数中α,γ等为不同损失值的权重参数。

![image](https://github.com/user-attachments/assets/37fc1e94-18e2-4e7d-bed9-10682b7063e1)



## 部分结果及说明

本次实验，在200之内的时间迭代步骤就可以得到较好的对抗性结果。对于各项超参数，在[原作代码](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles)参数默认值的基础上有略微修改。(最后一张图的结果，表示由原始的标签得到最后的攻击标签，以及攻击标签的概率值)

> 这一张原始值为clock可能是ImageNet中没有相应的类别图像
![image](https://github.com/user-attachments/assets/2418dedc-b3c0-4f11-a003-4b2248477736)






![image](https://github.com/user-attachments/assets/2242c176-a914-4f2c-bd0e-106c4b90f1fe)



![image](https://github.com/user-attachments/assets/396b3ac5-db2a-44c5-8568-301a400a71e3)



![image](https://github.com/user-attachments/assets/b85cbe7e-6845-4e5e-bd9f-189e377beedf)





