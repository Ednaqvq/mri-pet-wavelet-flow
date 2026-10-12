# MRI→PET：四子带 Huber 小波残差 Flow

主方法：**F_multih_no_boundary**。输入为预处理后的三维 MRI；输出为配对 PET 的归一化强度估计。模型使用 MRI 图像条件，clinical_dim=0。

## 模型

1. MRI AAE 编码分支与两层步长卷积结构分支提取特征，经门控融合和三个残差块生成 Coarse 潜变量。
2. 潜空间 Flow 在训练集 PET 潜空间统计量确定的坐标中进行输运，PET 解码器输出锚点 PET。
3. 完整锚点冻结。MRI 与锚点分别经过一层三维 Haar，八子带拼接提供多尺度条件；残差 Flow 从零状态出发，用两步 Euler 生成八子带残差。
4. LHH、HLH、HHL、HHH 四子带采用 Huber 流匹配监督，其余四子带采用平方项。边界混合块采用相同子带规则。
5. 每步速度经 MRI 支持投影；逆 Haar 后乘 0.1、叠加锚点，最终在 MRI 支持区域外置零。

训练损失包含流匹配、0.25 倍单次端点 L1、两步展开 L1、0.2 倍前景 SSIM 损失。两项 L1 除以残差尺度 0.1。Huber 阈值由源训练集锚点残差的子带 RMS 确定。

## 环境

验证环境为 Linux、Python 3.10、CUDA GPU，依赖版本见 `requirements.txt`。PyTorch 请安装与机器 CUDA 环境匹配的 2.10.0 构建。

```bash
python -m pip install -r requirements.txt
```

默认 BF16 混合精度，需要支持 BF16 的 GPU。推理入口启用 PyTorch 确定性算法、确定性 cuDNN 和 CUBLAS 工作区设置，保证同一环境下的重复输出稳定。运行时固定 num_workers=0、CPU 线程8、microbatch=1、梯度累积4。命令通过 `CUDA_VISIBLE_DEVICES` 选择一张空闲显卡；不同种子可分别运行在不同显卡上。

## 权重与推理

Release `v1.0.0` 提供 ADNI、NACC 各五个固定种子：42、2026、3407、1234、5678。每套包含 `anchor.pt`、`residual.pt` 与 SHA-256 清单。配套锚点校验值保存在残差权重中。

```bash
python download_weights.py --source ADNI --seed 42
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --weights weights/ADNI_seed42 \
  --mri /data/preprocessed_mri.nii.gz \
  --output runs/example/generated_pet.nii.gz
```

MRI 需要完成配准、去颅骨和与训练相同的强度处理，脑外为0、强度在[-1,1]。输入应与训练数据采用相同的 MNI 空间及轴顺序。入口将影像三线性重采样到160×196×160，并更新输出仿射矩阵以对应这一网格。输出为训练时的 PET 强度尺度。真实 PET 仅在训练与评价时使用。

`--source` 表示权重的训练来源。ADNI→NACC 使用 ADNI 权重和 NACC MRI，NACC→ADNI 使用 NACC 权重和 ADNI MRI。

## 从头训练

准备配对 MRI/PET 和一个 CSV：

```csv
subject_id,sample_id,t1_path,fdg_pet_path,has_fdg_pet,split,fold
subject_a,scan_a,subject_a/mri.nii.gz,subject_a/pet.nii.gz,True,train,1
subject_b,scan_b,subject_b/mri.nii.gz,subject_b/pet.nii.gz,True,train,0
subject_c,scan_c,subject_c/mri.nii.gz,subject_c/pet.nii.gz,True,final_val,-1
```

train/fold≠0 为训练集，train/fold=0 为验证集，final_val 为测试集。相对影像路径相对于 `--dataset-root`，三个集合的受试者互斥。MRI/PET 必须同网格且完成强度预处理。原实验规模为 ADNI 913/229/123、NACC 255/73/36 扫描。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --metadata /data/ADNI/metadata.csv --dataset-root /data/ADNI \
  --cohort ADNI --seed 42 --run-dir runs/ADNI_seed42
```

训练依次运行 MRI AAE、PET AE、训练集潜空间统计、Coarse与Flow联合训练、冻结Coarse后新初始化Flow、锚点缓存与训练集残差阈值、小波残差Flow。AE最多100轮；两轮潜空间Flow各最多200轮；残差最多60轮。学习率、提前停止和损失设置位于 `configs/default.json` 及 `src/residual_pipeline.py`。

相同命令支持从已提交的检查点继续训练；数据、配置和源码哈希会验证。写入 `<run-dir>/campaign/STOP_REQUESTED` 后，流程在检查点边界停止；继续前移走该文件。`--smoke` 在独立的 `preflight/` 下用每个集合两例、每阶段一轮检验整个训练链路。

主方法训练产物：

- `<run-dir>/frozen/fold_0/flow/best.pt`：完整锚点。
- `<run-dir>/F_multih_no_boundary/best.pt`：小波残差 Flow。
- `<run-dir>/anchor_config.json`：锚点配置。

原始训练检查点保留优化器和训练状态。公开 Release 提供模型参数、标准化统计量和推理配置的精简配对版本。导出方式见 `export_weights.py`。

```bash
python export_weights.py --run-dir runs/ADNI_seed42 --output weights/new_ADNI_seed42
```

## 同域与跨域评价

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --weights weights/ADNI_seed42 --metadata /data/NACC/metadata.csv \
  --dataset-root /data/NACC --output-dir runs/ADNI_to_NACC_seed42
```

输出逐扫描指标与受试者等权汇总。MAE/SSIM/PSNR 分别计算 MRI 前景与全体积；data_range=2；全体积指标使用最终支持约束后的 PET。五种子均值和样本方差使用 ddof=1。验证集选择沿用实验训练期间的前景 MAE。

## 文件导航

| 文件 | 作用 |
|---|---|
| `train.py` | 可迁移的完整训练入口与数据/配置校验 |
| `inference.py` | 配对权重加载、MRI推理、NIfTI保存 |
| `evaluate.py` | 同域/跨域评价 |
| `src/pipeline.py` | 多阶段训练衔接、冻结校验 |
| `src/mlrd/fusion_flow.py` | 双分支锚点与潜空间Flow |
| `src/mlrd/models.py` | AAE、门控Coarse、条件U-Net |
| `src/supported_flow.py` | 小波残差Flow与支持投影 |
| `src/residual_pipeline.py` | 主方法训练与图像端点损失 |
| `src/multih_huber_loss.py` | 四子带Huber监督 |
| `src/wave_tools.py` | Haar正变换、逆变换和子带指标 |
| `src/output_support.py` | 最终PET背景约束 |
| `validation/` | 发布前验证记录 |

影像与逐扫描受试者数据由使用者在自己的数据环境中准备。发布包包含代码、模型参数和不含病例标识的验证记录。
