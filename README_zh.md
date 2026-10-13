[English](README.md) | **中文**

# AKI 预测：考虑 storetime 的连续监测研究

> **数据准备：** 本仓库有意不包含课程提供的 MIMIC-IV 压缩包。复现之前请把 `MIMIC_IV.zip` 复制到 `data/raw/MIMIC_IV.zip`。见 [data/README.md](data/README.md)。

## 概述

本仓库是一项改编自 MIMIC-IV 的急性肾损伤（AKI）预测研究的干净、可复现的发布版本。它复现了五种基线模型架构，在模拟的连续监测下评估它们冻结后的预测，审计化验结果可用时间的问题，并对冻结模型做一次考虑 storetime 的压力测试。

## 研究问题

回顾性的连续 AKI 预测评估里，用化验的 `charttime` 而不是结果可用时间的代理（`storetime`），会不会实质性地影响评估结果？

## 科学流程

1. [第 1 步——基线复现](Step1_Baseline_Reproduction/README.md)：构建修正后的 MIMIC-IV 队列，在 0、48、72 小时三个预测时域上训练 5 个模型。
2. [第 2 步——连续监测](Step2_Continuous_Monitoring/README.md)：冻结 15 个正式模型、预处理、阈值、划分和队列，在七个监测时间点上打分。
3. [第 3 步——数据可用性审计](Step3_Data_Availability_Audit/README.md)：量化化验 `charttime` 和 `storetime` 之间的错位，不做推理、不重训。
4. [第 4 步——考虑 storetime 的压力测试](Step4_Storetime_Aware_Stress_Test/README.md)：重复冻结模型的推理，但化验行同时要求 `charttime <= cutoff` 和 `storetime <= cutoff`。
5. [结果](Result/README.md)：一组精简的最终表格和图。

## 主要发现

化验结果可用时间的错位相当可观，但考虑 storetime 的冻结模型压力测试表明，对大多数被评估的模型，判别性能几乎没有实质变化。

这个发现有意保持克制。`storetime` 只是结果可用时间的代理，不是临床医生看到结果的确切时刻。这次压力测试不能证明不存在时间泄漏，不能证明时间从不重要，也不能证明其他 EHR 模型都是稳健的。

## 仓库结构

```text
References/                         参考文献
data/raw/                           本地压缩包位置（Git 忽略）
Step1_Baseline_Reproduction/        修正后的静态基线流程与产物
Step2_Continuous_Monitoring/        冻结模型的连续监测评估
Step3_Data_Availability_Audit/      charttime / storetime 数据审计
Step4_Storetime_Aware_Stress_Test/  冻结模型压力测试
Result/                             精简的、已核验的结果
external/                           固定版本的上游官方代码工作树
work/                               重新生成的本地输出（Git 忽略）
```

## 数据准备

本仓库有意不包含课程提供的 MIMIC-IV 压缩包。

把课程提供的：

```text
MIMIC_IV.zip
```

复制到：

```text
data/raw/MIMIC_IV.zip
```

修正后的可执行流程只用这一个压缩包。`physionet.org.zip` 不是正式流程的要求。如果想把压缩包放在别处，在 PowerShell 里设一个环境变量：

```powershell
$env:MIMIC_IV_ZIP="D:\somewhere\MIMIC_IV.zip"
python scripts/check_data.py
```

这个脚本从不下载数据，也不联网访问 PhysioNet。

## 环境准备

验证过的环境是 Python 3.11.0、PyTorch 2.8.0+cu128、CUDA 12.8。Python 包的精确版本列在 `requirements.txt`。

15 个正式 checkpoint 的二进制文件用 Git LFS 存放。克隆或检出之前先安装 Git LFS，然后每个用户账号执行一次 `git lfs install`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_data.py
```

提取、审计、逻辑回归和大部分报告用 CPU 就够。XGBoost 可以在 CPU 上跑。LSTM-Attention、Masked CNN、ITE-Transformer 和反复的连续推理强烈建议用 CUDA。

## 复现

所有命令在仓库根目录下运行。脚本按自己所在位置解析路径，不依赖当前工作目录。体积大的再生成产物写到 `work/`，可以用 `AKI_WORK_DIR` 改到别处。

### 第 1 步

```powershell
python Step1_Baseline_Reproduction/scripts/prepare_mimic_events.py
python Step1_Baseline_Reproduction/scripts/build_mimic_cohort.py
python Step1_Baseline_Reproduction/scripts/run_all_baselines.py --scope full --horizons 0 48 72 --device cuda
python Step1_Baseline_Reproduction/scripts/finalize_model_artifacts.py
python Step1_Baseline_Reproduction/scripts/summarize_baselines.py --scope full
python Step1_Baseline_Reproduction/scripts/validate_baselines.py
python Step1_Baseline_Reproduction/scripts/build_artifact_manifest.py
```

### 第 2 步

```powershell
python Step2_Continuous_Monitoring/scripts/run_monitoring.py --stage dry-run
python Step2_Continuous_Monitoring/scripts/run_monitoring.py --stage full
```

内存紧张的机器可以每个进程只跑一个模型 / 时域组合，再用 `merge_monitoring_runs.py` 合并。

### 第 3 步

```powershell
python Step3_Data_Availability_Audit/scripts/01_extract_storetime.py
python Step3_Data_Availability_Audit/scripts/02_delay_distribution.py
python Step3_Data_Availability_Audit/scripts/03_timepoint_pending.py
python Step3_Data_Availability_Audit/scripts/04_onset_creatinine.py
python Step3_Data_Availability_Audit/scripts/05_figures_and_summary.py
```

### 第 4 步

必须先通过的零对照门槛和逐个执行的冻结推理命令，见 [Step4_Storetime_Aware_Stress_Test/README.md](Step4_Storetime_Aware_Stress_Test/README.md)。

## 结果

核验过的聚合结果在 [Result](Result/README.md)。不包含任何 MIMIC 原始数据或患者级预测。

## 参考文献

见 [References/README.md](References/README.md)。

## 上游代码

官方仓库 `https://github.com/Jovinus/aki-timeseries-continuous-monitoring` 以工作树快照的形式包含在内，对应提交 `6794acb7f615b5864171f8aa7c8ae610049cb947`。其内嵌的 `.git` 目录有意去掉了。见 [external/README.md](external/README.md)。

## 许可

- 本项目自己写的代码按 [MIT 许可](LICENSE)发布。
- `References/` 下的论文保留各自原出版许可；本项目的 MIT 许可不覆盖它们。
- `external/` 下的代码保留上游项目的原始许可；本项目的 MIT 许可不替代、不扩展那份许可。

## 可复现性说明

- 公开的基线是把尿量行限定在各自 ICU 停留期内（`intime <= charttime <= outtime`）之后的修正运行。
- 冻结的测试队列含 2,168 次停留、124 例 AKI；规范 SHA-256：`d56d72c99d71e56e5144b30f7f0a4238f670dad4a24dd739d3ef3b13c7222723`。
- 恰好包含 15 个正式 checkpoint。非正式和过时的模型都不包含。
- 重跑第 1 步会把模型和患者级产物写到被忽略的 `work/` 下，不会覆盖已发布的 checkpoint 快照。

## 局限

这是一项改编自 MIMIC-IV 的复现，不是对原论文私有 NHIS 开发环境的精确数值复刻。研究是单数据集、回顾性的。`storetime` 只是结果可用时间的代理，而且冻结模型的设计不能估计在考虑可用时间的输入上重新训练之后的性能。
