# Adversarial Camouflage Utilizing style Transfer

## 介绍

此次实验是对[Adversarial Camouflage: Hiding Physical-World Attacks with Natural Styles](https://arxiv.org/abs/2003.08757)思想的实现，并且只针对**数字攻击**。实验参考了[原作代码](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles)。我们使用的模型与论文中一致，为`torchvision.models`中预训练的`vgg19`，这是在`ImageNet`上预训练好的模型。

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
|-- Result.py
```

### Run Attack

直接运行main.py，使用不同的图像路径，以及相应各种的参数在main中修改

> main.py

## 方法

### 对抗性:

实现了两种形式的对抗，一种有明确目标标签，另一无目标标签。

![image](https://github.com/user-attachments/assets/ee66d33d-fa0d-4f0d-b569-c316ecf52187)



### 风格迁移:

为了使得对抗性样本与风格图像相似，通过对网络中的对应层特征对比的方法设计损失函数。G 是从特征提取器 F 中在第 l 层提取的特征的 Gram 矩阵。实验中使用了5层不同的特征，分别是conv1_1,conv2_1,conv3_1,conv4_1,conv5_1,详见实验代码。

![image](https://github.com/user-attachments/assets/95ca73ea-709c-485e-a3a8-a9bded144c8e)


### 内容一致:

为了保持对抗性干扰隐蔽性，对抗性样本不能过度偏离原始图像，这里仍然使用某一网络层得到的特征进行一致性度量，实验中使用了conv4_2得到的特征。

![image](https://github.com/user-attachments/assets/7f3d0515-ad90-44b0-ade6-fb99a134193d)


### 损失函数:

最终的损失函数中α,γ等为不同损失值的权重参数。

![image](https://github.com/user-attachments/assets/dc8f4099-2e67-4516-9727-85be714958f9)


## 部分结果及说明

本次实验，在200之内的时间迭代步骤就可以得到较好的对抗性结果。对于各项超参数，在[原作代码](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles)参数默认值的基础上有略微修改。(最后一张图的结果，表示由原始的标签得到最后的攻击标签，以及攻击标签的概率值)

![image](https://github.com/user-attachments/assets/2418dedc-b3c0-4f11-a003-4b2248477736)

> 这一张原始值为clock可能是ImageNet中没有相应的类别图像



![image](https://github.com/user-attachments/assets/2242c176-a914-4f2c-bd0e-106c4b90f1fe)



![image](https://github.com/user-attachments/assets/396b3ac5-db2a-44c5-8568-301a400a71e3)



![image](https://github.com/user-attachments/assets/b85cbe7e-6845-4e5e-bd9f-189e377beedf)





