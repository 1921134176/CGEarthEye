<div align="center">

# <img width="60" alt="image" src="assets/gaofen.png"> CGEarthEye: 吉林一号大模型构建与应用

<div align="center">
  <img width="300" alt="image" src="assets/logo.png">
  <br>
</div>

[\[🚀 Quick Start\]](https://www.jl1mall.com/) [\[📖 Report\]](./report) [\[📹 Weight\]](https://pan.baidu.com/s/12bds0ZTMwyRVgv7Nkq51Aw?pwd=cgwx)

![CGEarthEye](assets/model.png)

</div>

## Update 🚀🚀🚀

- 2025.05.30-CGEarthEye发布吉林一号亚米级光学遥感影像预训练权重。

## 介绍

为提升吉林一号遥感卫星影像应用的智能化水平，解 决视觉大模型在高分辨率卫星遥感影像上性能受限问题，我们构建了包含21亿参数量的吉林一号遥感大模型——CGEarthEye。CGEarthEye结合了生成式与对比式自监督学习算法的优势，具备对遥感影像全局与局部建模能力，并利用了全球分布的1500万高质量亚米级吉林一号卫星遥感影像样本，在16张A800 GPU上进行了训练。对比视觉领域大模型，CGEarthEye仅仅微调解码器的情况下，各项遥感任务显著优于全量微调的视觉领域大模型。对比遥感领域大模型，CGEarthEye具备大多数遥感领域大模型不具备的冻结微调能力，极大缩短应用微调时间与显存，缓解了大模型下游微调困难问题，并在4项任务10个数据集上实现冻结性能SOTA。

## 骨干

[Baidu](https://pan.baidu.com/s/12bds0ZTMwyRVgv7Nkq51Aw?pwd=cgwx)

|       模型       | 层数 | 编码维度 | 隐藏层维度 | 注意力头 | 参数量/M |
| :--------------: | :--: | :------: | :--------: | :------: | :------: |
| CGEarthEye-Small |  12  |   384    |    1536    |    6     |    22    |
| CGEarthEye-Base  |  12  |   768    |    3072    |    12    |    86    |
| CGEarthEye-Large |  24  |   1024   |    4096    |    16    |   307    |
| CGEarthEye-Huge  |  32  |   1280   |    5120    |    16    |   632    |
| CGEarthEye-Giant |  40  |   1536   |    6144    |    24    |   1100   |

## 测试

![experiments](assets/experiments.png)

## 应用

在应用方面，基于CGEarthEye，我们微调了20种应用模型，已上线吉林一号网[https://www.jl1mall.com/](吉林一号网)。

<div align="center">
  <img width="500" alt="image" src="assets/application.png">
  <br>
</div>

## 微调

最后，我们为吉林一号数据集编写了训练配置，用户可根据业务快速利用CGEarthEye完成应用的迭代。

### 环境


```bash
conda create -n CGEarthEye python=3.10
conda activate CGEarthEye
pip install -r requirements.txt
```
### 数据准备

#### 场景分类

- [AID](captain-whu.github.io/AID/)
- [NWPU_RESISC45](https://gcheng-nwpu.github.io/#Datasets)

```bash
|-datasets/SceneClassification
|----AID
|    |---Airport
|        |---airport_1.jpg
|        |---airport_2.jpg
|        |---    ···
|    |---BareLand
|        |---bareland_1.jpg
|        |---    ···
|    |---  ···
|    |---train_80per.txt
|    |---val_20per.txt
|----NWPU_RESISC45
|    |---airplane
|        |---airplane_001.jpg
|        |---    ···
|    |---airport
|        |---airport_0011.jpg
|        |---    ···
|    |---  ···
|    |---train_80per.txt
|    |---val_20per.txt
...
```

#### 语义分割

- [ISAID](https://captain-whu.github.io/iSAID/index.html)
- [cropland](http://10.200.30.48:8080/store)

```bash
|-datasets/SemanticSegmentation
|----cropland
|    |---train
|        |---Img
|            |---LC_01_000001.png
|            |---LC_01_000002.png
|            |---    ···
|        |---Label
|            |---LC_01_000001.png
|            |---LC_01_000002.png
|            |---    ···
|    |---val
|        |---Img
|            |---LC_01_000001.png
|            |---LC_01_000002.png
|            |---    ···
|        |---Label
|            |---LC_01_000001.png
|            |---LC_01_000002.png
|            |---    ···
|    |---test
|        |---Img
|            |---LC_01_000001.png
|            |---LC_01_000002.png
|            |---    ···
|        |---Label
|            |---LC_01_000001.png
|            |---LC_01_000002.png
|            |---    ···

|----ISAID
|    |---img_dir
|        |---train
|            |---P0003_0_896_0_896.png
|            |---    ···
|        |---val
|            |---P0003_0_896_0_896.png
|            |---    ···
|        |---test
|            |---P0003_0_896_0_896.png
|            |---    ···
|    |---ann_dir
|        |---train
|            |---P0003_0_896_0_896_instance_color_RGB.png
|            |---    ···
|        |---val
|            |---P0003_0_896_0_896_instance_color_RGB.png
|            |---    ···
|        |---test
|            |---P0003_0_896_0_896_instance_color_RGB.png
|            |---    ···
...
```

#### 变化检测
- [LEVIR-CD](https://opendatalab.com/OpenDataLab/LEVIR-CD)
- [SYSU-CD](https://github.com/liumency/SYSU-CD)
- [CDD](https://paperswithcode.com/dataset/cdd-dataset-season-varying)
```bash
|-datasets/ChangeDetection
|----SYSU-CD
|    |---train
|        |---Image1
|           |---00000.png
|           |---    ···
|        |---Image2
|           |---00000.png
|           |---    ···
|        |---Label
|           |---00000.png
|           |---    ···
|    |---val
|        |---Image1
|           |---00000.png
|           |---    ···
|        |---Image2
|           |---00000.png
|           |---    ···
|        |---Label
|           |---00000.png
|           |---    ···
|    |---test
|        |---Image1
|           |---00000.png
|           |---    ···
|        |---Image2
|           |---00000.png
|           |---    ···
|        |---Label
|           |---00000.png
|           |---    ···
|----LEVIR-CD
|...
|----CDD
|...
```
#### 目标检测
- [DIOR / DIOR-R](www.escience.cn/people/JunweiHan/DIOR.html)


```bash
|-datasets/ObjectDetection
|----DIOR
|    |---images
|        |---trainval/
|        |    |---00001.jpg
|        |    |---00002.jpg
|        |    |---  ...
|        |---test/
|        |    |---11726.jpg
|        |    |---11727.jpg
|        |    |---  ...
|    |---Annotations
|        |---trainval.json
|        |---test.json

|----DIOR-R
|    |---images
|        |---trainval/
|        |    |---00001.jpg
|        |    |---00002.jpg
|        |    |---  ...
|        |---test/
|        |    |---11726.jpg
|        |    |---11727.jpg
|        |    |---  ...
|    |---ImageSets
|        |---Main
|        |    |---train.txt
|        |    |---val.txt
|        |    |---test.txt
|    |---Annotations
|        |---Oriented Bounding Boxes
|        |    |---00001.xml
|        |    |---00002.xml
|        |    |---  ....
```


### 模型训练

#### 场景分类

```bash
# 单机单卡
python tools/train_sc.py \
    config/SceneClassification/CGEarthEye-Giant-518-AID.py \
    --amp
```
```bash
# 单机多卡
python tools/dist_train_sc.sh \
    config/SceneClassification/CGEarthEye-Giant-518-AID.py
```
#### 语义分割

```bash
# 单机单卡
python ./tools/train_ss.py config/SemanticSegmentation/CGEarthEye-Giant-518-ISAID
```
```bash
# 单机多卡
bash ./tools/dist_train_ss.sh config/SemanticSegmentation/CGEarthEye-Giant-518-ISAID 4
```
#### 变化检测

```bash
# 单机单卡
python ./tools/train_cd.py config/ChangeDetection/CGEarthEye-Giant-518-levircd.py
```
```bash
# 单机多卡
bash ./tools/dist_train_cd.sh config/ChangeDetection/CGEarthEye-Giant-518-levircd.py 4
```
#### 目标检测

```bash
# 水平框检测（MMDetection3.x）
## 单机单卡
python tools/train_hbb.py config/ObjectDetection/HBB/CGEarthEye-Giant-784-DIOR.py
## 单机多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train_hbb.sh config/ObjectDetection/HBB/CGEarthEye-Giant-784-DIOR.py 4
```

```bash
# 旋转框检测（MMRotate1.x）
## 单机单卡
python tools/train_obb.py config/ObjectDetection/OBB/CGEarthEye-Giant-784-DIORR.py
## 单机多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train_obb.sh config/ObjectDetection/OBB/CGEarthEye-Giant-784-DIORR.py 4
```

### 模型测试

#### 场景分类

```bash
# 单机单卡
python tools/test_sc.py \
    config/SceneClassification/CGEarthEye-Giant-518-AID.py
```
```bash
# 单机多卡
python tools/dist_test_sc.sh \
    config/SceneClassification/CGEarthEye-Giant-518-AID.py
```

#### 语义分割
```bash
# 单机单卡
python tools/test_ss.py \
    config/SemanticSegmentation/CGEarthEye-Giant-518-ISAID.py
```
```bash
# 单机多卡
python tools/dist_test_ss.sh \
    config/SemanticSegmentation/CGEarthEye-Giant-518-ISAID.py
```

#### 变化检测
```bash
# 单机单卡
python ./tools/test_cd.py config/ChangeDetection/CGEarthEye-Giant-518-levircd.py work_dir/last.pth
```
#### 目标检测

```bash
# 水平框检测（MMDetection3.x）
## 单机单卡
python tools/test_hbb.py \
    config/ObjectDetection/HBB/CGEarthEye-Giant-784-DIOR.py \
    Path/To/Your/Weight/hbb_model.pth
## 单机多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test_hbb.sh \
    config/ObjectDetection/HBB/CGEarthEye-Giant-784-DIOR.py \
    Path/To/Your/Weight/hbb_model.pth 4
```

```bash
# 旋转框检测（MMRotate1.x）
## 单机单卡
python tools/test_obb.py \
    config/ObjectDetection/OBB/CGEarthEye-Giant-784-DIORR.py \
    Path/To/Your/Weight/obb_model.pth
## 单机多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test_obb.sh \
    config/ObjectDetection/OBB/CGEarthEye-Giant-784-DIORR.py \
    Path/To/Your/Weight/obb_model.pth 4
```

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## 💡 Relevant Projects

[1] <strong>Skysense: A multi-modal remote sensing foundation model towards universal interpretation for earth observation imagery, IEEE CVPR, 2024</strong> | [Paper](https://arxiv.org/abs/2312.10115) 
<br><em>&ensp; &ensp; &ensp;Xin Guo<sup>&#8727;</sup>, Jiangwei Lao<sup>&#8727;</sup>, Bo Dang, Yingying Zhang, Lei Yu,Lixiang Ru,Liheng Zhong,Ziyuan Huang,Kang Wu,Dingxiang Hu,Huimei He,Jian Wang,Jingdong Chen,Ming Yang,Yongjun Zhang and Yansheng Li</em>

[2] <strong>Mtp: Advancing remote sensing foundation model via multi-task pretraining, IEEE JSTARS, 2024</strong> | [Paper](https://arxiv.org/abs/2403.13430/) | [Github](https://github.com/ViTAE-Transformer/MTP)
<br><em>&ensp; &ensp; &ensp;Di Wang<sup>&#8727;</sup>, Jing Zhang<sup>&#8727;</sup>, Minqiang Xu<sup>&#8727;</sup>, Lin Liu, Dongsheng Wang, Erzhong Gao,Chengxi Han,Haonan Guo and Bo Du</em>

[3] <strong>DINOv2: Learning Robust Visual Features without Supervision,2024</strong> | [Paper](arxiv.org/abs/2304.07193) | [Github](github.com/facebookresearch/dinov2)
<br><em>&ensp; &ensp; &ensp;Maxime Oquab<sup>&#8727;</sup>, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin and Piotr Bojanowski</em>
