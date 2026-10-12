# 正式实验固定划分

这两份 CSV 来自 MRI-only 正式实验的 `subject_safe_split.csv`，与发布的 ADNI、NACC 五种子权重对应。文件内容及行顺序保留原样。

| 数据集 | 训练扫描 / 受试者 | 验证扫描 / 受试者 | 测试扫描 / 受试者 | 总扫描 / 受试者 |
|---|---:|---:|---:|---:|
| [ADNI](ADNI_split.csv) | 913 / 442 | 229 / 111 | 123 / 45 | 1265 / 598 |
| [NACC](NACC_split.csv) | 255 / 255 | 73 / 73 | 36 / 36 | 364 / 364 |

## 字段与归属

- `subject_id`：数据集受试者编号。
- `sample_id`：扫描编号，与本地 MRI/PET 配对表连接。
- `split`、`fold`：原实验划分字段，读取规则如下。

| 实际用途 | split | fold |
|---|---|---|
| 训练集 | train | 非0 |
| 验证集 | train | 0 |
| 测试集 | final_val | -1 |

ADNI 训练集合并原始 fold 1、2、3、4；NACC 训练集为 fold 1。42、2026、3407、1234、5678 五个种子改变模型训练的随机初始化和采样，均使用同一套训练、验证及测试样本。

同域测试使用该源数据集的测试集。ADNI→NACC 使用 ADNI 训练的模型评价 NACC 测试集；NACC→ADNI 使用 NACC 训练的模型评价 ADNI 测试集。

## 连接影像路径表

按 `subject_id`、`sample_id` 将划分表与本地影像路径表一对一连接，得到 `train.py` 和 `evaluate.py` 使用的 metadata CSV。影像路径表提供 `t1_path`、`fdg_pet_path`、`has_fdg_pet`，划分表提供 `split`、`fold`。应检查划分表每一行均成功匹配，编号列按字符串读取。

## 校验

`verification.json` 记录原始 metadata 和划分 CSV 的 SHA-256、扫描及受试者计数，以及五个种子的权重配对校验结果。已逐行核对划分 CSV 与实际训练 metadata 的四列投影一致，10套锚点权重记录的 metadata 哈希均匹配。三个集合的受试者互斥，扫描编号唯一。
