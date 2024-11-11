# Adversarial Camouflage Utilizing style Transfer

## 介绍

此次实验是对[Adversarial Camouflage: Hiding Physical-World Attacks with Natural Styles](https://arxiv.org/abs/2003.08757)思想的实现，并且只针对**数字攻击**。实验参考了[原作代码]([RjDuan/AdvCam-Hide-Adv-with-Natural-Styles: Code for "Adversarial Camouflage: Hiding Physical World Attacks with Natural Styles" (CVPR 2020)](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles))。我们使用的模型与论文中一致，为`torchvision.models`中预训练的`vgg19`，这是在`ImageNet`上预训练好的模型。

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

|--data
|  |--content
|    |--stop-sign
|  |--style
|    |--stop-sign
|  |--content-mask
|  |--style-mask

|--Vgg
|  |--Vgg19.py

|--main.py

|--attack.py

|--loss.py

|--utils.py

|--Result.py

### Run Attack

直接运行main.py，使用不同的图像路径，以及相应各种的参数在main中修改

> main.py

## 方法

### 对抗性:

实现了两种形式的对抗，一种有明确目标标签，另一无目标标签。

**目标攻击：**

使得分类结果偏离真实结果$y_{true}$，并且接近目标标签$y_t$
$$
\mathcal{L}_{adv}=\beta\mathcal{L}_{ce}(x_{adv},y_t)+\mathcal{L}_{ce}(x_{adv},y_{true})
$$
**非目标攻击：**
$$
\mathcal{L}_{adv}=-\mathcal{L}_{ce}(x_{adv},y_{true})
$$

### 风格迁移:

为了使得对抗性样本与风格图像相似，通过对网络中的对应层特征对比的方法设计损失函数。$G$ 是从特征提取器 $F$ 中在第 $l$ 层提取的特征的 Gram 矩阵。实验中使用了5层不同的特征，分别是conv1_1,conv2_1,conv3_1,conv4_1,conv5_1,详见实验代码。
$$
\mathcal{L}_{style} = \sum_{l} \left\| G(F_l(x_{adv})) - G(F_l(x_s)) \right\|^2
$$

### 内容一致:

为了保持对抗性干扰隐蔽性，对抗性样本不能过度偏离原始图像，这里仍然使用某一网络层得到的特征进行一致性度量，实验中使用了conv4_2得到的特征。
$$
\mathcal{L}_{content} = \sum_{l \in L_{content}} \left\| F_l(x_{adv}) - F_l(x) \right\|^2
$$

### 损失函数:

最终的损失函数中$\lambda_{adv},\alpha,\gamma$为不同损失值的权重参数。
$$
\mathcal{L}_{total} = \lambda_{adv} \cdot \mathcal{L}_{adv} + \alpha \cdot \mathcal{L}_{style} + \gamma \cdot \mathcal{L}_{content}
$$

## 部分结果及说明

本次实验，在200之内的时间迭代步骤就可以得到较好的对抗性结果。对于各项超参数，在[原作代码]([RjDuan/AdvCam-Hide-Adv-with-Natural-Styles: Code for "Adversarial Camouflage: Hiding Physical World Attacks with Natural Styles" (CVPR 2020)](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles))参数默认值的基础上有略微修改。(最后一张图的结果，表示由原始的标签得到最后的攻击标签，以及攻击标签的概率值)

![1731317366205](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\1731317366205.png)

> 这一张原始值为clock可能是ImageNet中没有相应的类别图像



![1731319936530](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\1731319936530.png)



![1731321305027](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\1731321305027.png)



![1731327111001](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\1731327111001.png)




