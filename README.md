# 北大选课网验证码识别 2021 年春季学期

Powered by *Elector Quartet* (***@xmcp**, @SpiritedAwayCN, @Rabbit, @gzz*)

## 数据集描述

最初的数据集为 5130 张人工标记的验证码，之后利用早期训练好的模型在选课网上进行自动验证 (自举)，又收集到 102741 张验证码，并对其中 2363 张识别错误的验证码进行人工标注。以上三部分合并为 GIF 验证码数据集

对 GIF 验证码数据集进行预处理，得到实际用于模型训练的数据集，它由裁剪好的单字母灰度图组成，图片尺寸 52x52，8 位色深，文件格式 PNG

训练集/测试集按照 8:2 进行划分，对训练集的数据进行数据增强，方式为旋转特定的几个角度，测试集不做特殊处理

在 const.py 中，`DATA_RAW_DIR` 对应最初的 5130 张人工标记的验证码，`DATA_BOOTSTRAP_DIR` 对应所有通过自举得到的验证码 (包含验证正确/错误的验证码)，`DATA_MANUALLY_LABELED_DIR` 对应所有人工标注的验证码

## 数据集下载

百度网盘下载链接：[https://pan.baidu.com/s/12GcfXYaLbHFC1WkzWhre2w](https://pan.baidu.com/s/12GcfXYaLbHFC1WkzWhre2w) (提取码：xsl3)

1. 最初的 5130 张人工标注的验证码 `xmcp_5130.zip`，由 xmcp 提供，详见 [xmcp/elective-dataset-2021spring][ref-xmcp-dataset]
2. 自举得到的验证码以 10000 张为单位，分为 11 个 `bootstrap_all.x.zip`
3. 人工标注的验证码为 `manually_labeled.zip`，其中 `manually_labeled_20210309/` 对应第一批加入到 CNN 模型训练的人工标注验证码，`manually_labeled_20210310/` 和 `manually_labeled_20210311/` 对应第二批加入到 CNN 模型训练的人工标注验证码
4. 直接用于模型 210311_1 训练/测试的单字母灰度图 `crop.20210311.1.tar.gz`，共包含 354008 张单字母灰度图，对应于毫秒时间戳不超过 1615449512408 的 88502 张 GIF 验证码

## 图像预处理

基本采用 [SpiritedAwayCN/ElectiveCaptCha][ref-SpiritedAwayCN-ElectiveCaptCha] 的方法进行预处理，相关算法的介绍参看 [SpiritedAwayCN/ElectiveCaptCha][ref-SpiritedAwayCN-ElectiveCaptCha] 的说明

本项目对 [SpiritedAwayCN/ElectiveCaptCha][ref-SpiritedAwayCN-ElectiveCaptCha] 的算法做了以下两点改进：

1. 验证码的字母可能会变色，此时隔 4 帧作差，会有上一个字母的残留，因此加入 `M_mask` 的相关代码，每提取出一个字母，就将其加入到 `M_mask` 中，下一个字母提取的时候，会通过借助 `M_mask` 将之前已经提取出的字母对应的像素从当前字母的提取结果中抹去，以消除字母变色对后 3 个字母的提取所产生的影响
2. 验证码的第一个字母可能在前 4 帧里颜色较浅，取第 4 帧转为灰度图后，第一个字母对应像素的灰度值大于全局平均灰度值，利用 OSTU 二值化时会被抹去，填充成白色，最后得到一张只剩下少量噪音残留的空白图像。改进的方法是，首先将 4 个关键帧的灰度图叠加，保留颜色最深的灰度值 (最小值)，利用第 5-16 帧中第一个字母的颜色逐渐加深的特点，获得灰度值较深的包含第一个字母轮廓的 `M_merge`，此时再利用 OSTU 二值化，就基本上能留住第一个字母的信息，然后将 `M_merge` 的右三个字母抹去，仅保留包含第一个字母的轮廓，再根据此时的 `M_merge`，将第 1 个关键帧的灰度图中第一个字母对应像素的灰度值降低，此时再调用之前的处理函数，就可以更加准确地提取出第一个字母

## 模型训练与测试

以测试集的测试结果作为模型训练结果的参考，以模型在选课网上在线验证的结果作为模型的实际准确率

| 模型     | 测试集准确率 (单字母) | 在线测试准确率 (四字母) / % | 备注 |
| :------- | :------------------ | :------------------------ | :--- |
| 210308_3 | 0.9968 (15818 / 15869) | 0.9675 (17253 / 17832) | 扩大数据集，进行数据增强 |
| 210309_2 | 0.9952 (33811 / 33973) | 0.9799 (11659 / 11898) | 扩大数据集，加入第一批人工标注的验证码 |
| 210309_3 | 0.9951 (33807 / 33973) | 0.9788 (11847 / 12104) | 同 210309_2，但数据增强的旋转角度范围缩小至 [-15, 15] |
| 210309_4 | 0.9928 (33730 / 33973) | 0.9727 (9527 / 9794)   | 同 210309_2，但不进行数据增强 |
| **210311_1** | **0.9977** (70467 / 70630) | **0.9916** (10219 / 10306) | 扩大数据集，加入第二批人工标注的验证码，优化第一个字母的图像处理算法 |
| 210311_2 | 0.9961 (70643 / 70918) | 0.9838 (11230 / 11415) | 同 210311_1，但使用未经优化的图像处理算法 |

目前准确率最高的模型为 210311_1，该模型的训练结果、训练时的终端输出、训练/测试的相关代码已在 `dist/` 中提供

## 环境依赖

开发环境 Python 版本为 3.6.8，相关依赖包版本参考 requirements.txt，其中 PyTorch 版本应大于 1.4.x，否则无法读取 `dist/` 中提供的模型

模型训练在阿里云服务器上进行，CUDA 版本 11.0，使用的 GPU 为 Tesla V100-SXM2-32GB，使用的 CPU 为 Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz，训练集一次性读入内存，以 210311_1 的训练/测试为例，数据量为 88502 张 GIF 验证码，数据增强的旋转角度范围为 [-30, 30]，数据集扩增至 9 倍规模，占用的内存量为 32.9 GB，每个 Epoch 的耗时约为 309 秒

## 文件描述

### preprocess.py

图像预处理的相关算法，实际有用的函数是 `extract_c0_v1`, `extract_c0_v2`, `extract_c123`, `crop`，处理流程参考 `main` 函数

### cnn.py

与 CNN 模型相关的代码，包括数据集定义、数据增强、CNN 模型定义、训练集/测试集划分、模型训练、模型测试等

### bootstrap.py

与模型自举相关的代码，在选课网上在线测试现有模型的准确率，并且将预测正确/错误的模型分类收集，用于扩增数据集

### label_server.py

用于人工标注验证码的网站后端，前端代码在 `web/` 中。提供图像预处理、输入检查等功能，为人工标注验证码提供方便，所使用的数据源为模型自举过程中收集到的识别错误的验证码

### config.ini

与选课网账户相关的信息，包括学号、密码、是否为双学位账号，主要用于 `bootstrap.py`

## 致谢

- 最初的 5130 张人工标记的验证码来源于 xmcp 所提供的的数据集 [xmcp/elective-dataset-2021spring][ref-xmcp-dataset]
- 图像处理模块绝大部分参考了 [SpiritedAwayCN/ElectiveCaptCha][ref-SpiritedAwayCN-ElectiveCaptCha] 的 `preprocess_v3.py`
- 模型自举代码参考了 [SpiritedAwayCN/ElectiveCaptCha][ref-SpiritedAwayCN-ElectiveCaptCha] 的 `bootstrap.py`，该代码最初由 xmcp 提供
- 人工标注验证码的网站后端代码和前端设计参考了 xmcp 的相关工作

[ref-SpiritedAwayCN-ElectiveCaptCha]: https://github.com/SpiritedAwayCN/ElectiveCaptCh
[ref-xmcp-dataset]: https://github.com/xmcp/elective-dataset-2021spring
