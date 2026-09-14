# TriGlue：训练与预测

输入化合物SMILES及两条蛋白质序列，训练模型并对候选三元组评分。无需提供结合界面或三元复合物结构。蛋白图由单体PDB结构构建；可以提供已有结构，或使用预处理中的ESMFold生成缺失结构。

## 1. 安装

```bash
conda env create -f environment.yml
conda activate triglue
python scripts/smoke_test.py
```

使用Linux及所附环境。已有特征可用CPU预测。所需内存随序列长度变化。按[英文说明的安装部分](README.md#install)准备MoLFormer、ESM-2和Uni-Mol2，并设置`UNIMOL_WEIGHT_DIR`。使用下方自动生成结构的命令时，还需准备ESMFold；直接提供完整匹配的PDB结构时不需要它。

## 2. 数据

预测CSV需要`smiles,protein1_sequence,protein2_sequence`三列。训练还需要`triplet_id,label`；训练集和验证集的编号不重复，标签为0或1。文件使用UTF-8编码。

`examples/train.csv`和`examples/val.csv`是检查流程用的人工示例，标签无实验含义；正式使用时替换为自己的标注数据。

填写序列本身，不要填写蛋白名称。序列使用标准氨基酸字母且不超过1200个残基；化合物含氢原子数不超过256。可选的`protein1_structure`、`protein2_structure`填写PDB路径，结构需与完整序列匹配；此时可加`--structure-backend none`。

## 3. 生成特征

```bash
python data_process.py --input examples/train.csv --output-root features/train \
  --device cuda --fold-device cuda --fold-model-path models/esmfold
python data_process.py --input examples/val.csv --output-root features/val \
  --device cuda --fold-device cuda --fold-model-path models/esmfold
```

成功后得到`prepared_triplets.csv`。只检查输入格式时增加`--validate-only`。若输入已包含两个蛋白的PDB路径，将上述`--fold-device cuda --fold-model-path models/esmfold`替换为`--structure-backend none`。

## 4. 训练

```bash
python train.py --train features/train/prepared_triplets.csv \
  --val features/val/prepared_triplets.csv \
  --feature-root features/train --val-feature-root features/val \
  --molformer-model models/molformer --device cuda --output-dir results/run1
```

默认Adam、学习率1e-4、batch size 16、最多50轮；验证集Accuracy连续20轮未改善时停止。损失为BCE加0.1倍InfoNCE。训练使用15% SMILES token掩码（80/10/10）、10%图删边及σ=0.02的Uni-Mol2/ESM-2嵌入加噪；验证和预测不做增强。仅检查一次执行可增加`--epochs 1`，这会缩短默认训练时长。

每次使用新的输出目录。程序保存`best.pth`、`run_config.json`、训练记录及验证预测。

## 5. 预测

```bash
python predict.py --input features/val/prepared_triplets.csv --feature-root features/val \
  --checkpoint results/run1/best.pth --run-config results/run1/run_config.json \
  --molformer-model models/molformer --device cpu --output results/predictions.csv
```

新候选先按步骤3生成特征，再替换上述输入与特征目录。输出中的`inducibility_score_mean`为模型分数，`rank_within_protein_pair`为同一蛋白对内的候选排名；分数不是经校准的实验成功概率。

报错时检查提示中的行号或路径，以及`feature_build_summary.json`。常见原因包括模型文件缺失、输入不合法、结构与序列不匹配、显存不足。

## 6. 不确定性与可解释性

先按步骤4用不同的`--seed`训练多个模型，例如保存到`results/run1`和`results/run2`。每个模型的验证集预测保存在其目录下的`validation_predictions.csv`。另行按步骤3准备与验证集不重叠的测试集特征，保存到`features/test`，然后分别预测：

```bash
for run in run1 run2; do
  python predict.py --input features/test/prepared_triplets.csv --feature-root features/test \
    --checkpoint results/$run/best.pth --run-config results/$run/run_config.json \
    --molformer-model models/molformer --device cpu --output results/$run/test_predictions.csv
done
```

运行校准与不确定性分析，两个文件列表按同一模型顺序填写；更多模型可继续追加：

```bash
python calibration.py \
  --validation results/run1/validation_predictions.csv results/run2/validation_predictions.csv \
  --predictions results/run1/test_predictions.csv results/run2/test_predictions.csv \
  --output-dir results/calibration
```

程序仅用验证集标签拟合温度，结果保存到新的输出目录`results/calibration`：

| 文件 | 内容 |
|---|---|
| `calibrated_predictions.csv` | 原始及校准后的集成概率、校准后的模型间标准差`ensemble_sd`、分类置信度`calibrated_confidence`；有标签时还包括分类是否正确 |
| `temperatures.json` | 各模型和集成均值的温度参数 |
| `calibration_metrics.csv` | 校准前后的ECE、NLL、Brier score |
| `reliability_raw.svg`、`reliability_calibrated.svg` | 校准前后的可靠性图；对应分箱数据另存为同名CSV |
| `uncertainty_comparison.svg` | 正确与错误分类样本的预测标准差分布 |
| `confidence_comparison.svg` | 正确与错误分类样本的校准置信度分布 |

验证预测必须包含真实标签；测试预测包含`label`时生成全部评估结果。对无标签的新候选，只输出校准预测和温度参数。不确定性分析至少需要两个模型。

对单个候选执行原子/残基遮挡解释：

```bash
python explain.py --input features/val/prepared_triplets.csv --feature-root features/val \
  --triplet-id val_1 --checkpoint results/run1/best.pth --run-config results/run1/run_config.json \
  --molformer-model models/molformer --device cpu --output-dir results/explanation_val1
```

输出原子和残基重要性CSV、最多六个重要原子的高亮图及残基重要性图。`delta_logit`为遮挡前减遮挡后的logit，`abs_delta_logit`为其绝对值。原子遮挡将一个分子图节点特征置零；残基遮挡仅将一个one-hot位置置零，其他通道保持不变。
