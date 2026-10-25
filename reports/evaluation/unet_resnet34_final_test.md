# U-Net / ResNet34：最终测试结果

日期：2026-10-25。由项目作者在冻结权重和评价规则后运行，完整评价测试清单中的68张原图。评价规则见[评价方法](../../docs/评价方法.md)。

## 像素分割结果

| 全局指标 | 精确值 | 展示值 |
| --- | ---: | ---: |
| IoU | 0.7640134745977549 | 76.40% |
| Precision | 0.8799380054953743 | 87.99% |
| Recall | 0.8529266636367098 | 85.29% |
| F1 / Dice | 0.8662218124745013 | 86.62% |

累计像素TP=8,848,419、FP=1,207,311、FN=1,525,766。指标由全套图片累计计数计算，未按逐图指标取均值。

[结构化指标](unet_resnet34_final_test_metrics.json) · [终端输出](logs/unet_resnet34_final_test.txt)

## 模型与评价配置

- 模型：SMP U-Net / ResNet34，单通道logits。
- 检查点：epoch20，`outputs/run_20261024_181934_060182/crack_segmentation_model.pth`。
- 权重SHA-256：`0dbddc203bee5511a2e7b67438c09ca7b01b42305e986b8214ab36bdc6e6e692`。
- 清单：[test_ids.txt](../../data/test_ids.txt)，68张；训练/验证/测试为322/68/68，原图随机划分seed=1。
- 输入：EXIF方向规范化、RGB转float32/255和ImageNet标准化；标注灰度≥128为裂缝。
- 推理：原尺寸512滑窗、末窗贴边、重叠sigmoid概率平均、概率≥0.5为裂缝，CUDA、eval和无梯度模式。

对应运行入口：

```powershell
python -m scripts.evaluate --split test --model-dir "outputs/run_20261024_181934_060182"
```

## 图片级辅助判断

预测裂缝像素占全图≥0.003时判定有裂缝。测试清单68张均为阳性，图片级TP=68、FP=0、FN=0、TN=0。该结果只描述本批阳性样本检出情况，没有真实阴性样本可用于估计误报率。

外部六张弱标签样例的TP=2、FP=3、FN=1、TN=0，与这里的像素指标属于不同评价范围，详见[全部外部案例](external_presence_fresh_v1/report.md)。

## 范围说明

[同模型验证结果](unet_resnet34_validation_metrics.json)为IoU73.79%、Precision85.57%、Recall84.27%、F1 84.92%。验证集参与选模；本次测试来自不同图片，分数差不表示新增训练收益。随机原图划分的场景独立性尚未核验。

图像循环进度条耗时约76秒、约1.13秒/图，包含读取与计算，不含模型创建和加载。最终测试结果用于报告冻结版本的表现。

