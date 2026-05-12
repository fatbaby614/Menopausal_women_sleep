# MenoSCA-FBTS: 更年期女性特定睡眠分期算法
## Menopause-Specific Sleep Stage Classification Algorithm

[![Journal](https://img.shields.io/badge/Journal-Sleep%20Medicine-blue)](https://www.journals.elsevier.com/sleep-medicine)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-yellow)]()

---

## 📋 项目概述

本研究开发了一款专门针对更年期女性人群优化的自动化睡眠分期算法 MenoSCA-FBTS，通过融合黎曼几何多频段特征提取技术，系统研究了更年期女性人群的独特睡眠架构改变，并通过三大独立公共数据库实现结果交叉验证，确保研究结论的普适性和可靠性。

---

## 🔬 核心创新点

1. **三数据库独立交叉验证**
   - Sleep-EDF Database Expanded
   - ISRUC-Sleep Dataset
   - DREAMS Subjects Database
   
2. **6种SOTA算法全面基准对比**
   - YASA (87.0% Accuracy)
   - TinySleepNet (86.9% Accuracy, 轻量级深度学习)
   - Random Forest (86.7% Accuracy)
   - MenoSCA-FBTS (85.6% Accuracy)
   - EEGNet (81.0% Accuracy)
   - MiniRocket (72.0% Accuracy)

3. **关键临床发现**
   - 更年期女性深度睡眠(N3)占比显著降低
   - 更年期女性浅睡眠(N1)占比系统性升高
   - 睡眠状态不稳定性显著增加（Wake→N1过渡率提升2.5倍）

---

## 📁 项目目录结构

```
Menopausal_women_sleep/
├── cloudStudio/                          # 全部实验代码目录
│   ├── experiment_three_groups_paper.py   # 三群体统计分析主脚本
│   ├── experiment_sota_comparison.py     # 6种SOTA算法对比实验
│   ├── experiment_ablation_study.py      # 消融实验分析脚本
│   ├── sca_fbts_woman.py                 # MenoSCA-FBTS核心模型实现
│   ├── analysis_comprehensive.py         # 综合指标分析
│   ├── run_all_experiments.sh            # 一键批量运行全部7大实验
│   ├── experiment_results/                # 实验结果JSON/图库存放
│   └── results_pytorch/                  # PyTorch深度学习结果
├── Paper/                                # 论文相关资源
│   ├── els-cas-templates/
│   │   └── main.tex                       # 主论文LaTeX源文件
│   └── dataUsed/                          # 本次全部实验结果归档
│       ├── UPDATE_ARCHIVE.md              # 详细更新日志
│       └── UPDATE_SUMMARY.md              # 核心发现总结
└── README.md                              # 本说明文档
```

---

## 📚 数据集配置

请在代码中修改数据集路径指向你的本地数据存放位置：

| 数据集名称 | 环境变量(Windows) | 路径示例 |
|-----------|-----------------|---------|
| Sleep-EDF | SLEEP_EDF_DIR | E:\datasets\Sleep\sleep-edf-database-expanded-1.0.0 |
| ISRUC-Sleep | ISRUC_DIR | e:\datasets\Sleep\ISRUC-SLEEP |
| DREAMS | DREAMS_DIR | e:\datasets\Sleep\DREAMS |

---

## 🚀 快速开始

### 环境依赖安装
```bash
pip install numpy scipy scikit-learn xgboost mne matplotlib pandas
pip install torch braindecode
```

### 一键运行全部完整实验
```bash
cd cloudStudio
bash run_all_experiments.sh
```

### 单独运行某一个实验
```bash
# 三群体统计分析
python experiment_three_groups_paper.py --dataset sleep-edf

# 6种SOTA算法对比
python experiment_sota_comparison.py

# 消融实验
python experiment_ablation_study.py
```

---

## 📊 核心性能结果汇总

| 算法 | 准确率(%) | Macro F1(%) | 训练耗时(s) |
|------|----------|------------|------------|
| YASA | 87.0 ± 1.8 | 57.5 ± 3.2 | 153 |
| TinySleepNet | 86.9 ± 3.1 | 58.9 ± 4.5 | 933 |
| Random Forest | 86.7 ± 2.1 | 60.6 ± 3.6 | 20 |
| MenoSCA-FBTS | 85.6 ± 2.4 | 53.1 ± 4.2 | 71 |
| EEGNet (braindecode) | 81.0 ± 4.8 | 48.8 ± 6.2 | 862 |
| MiniRocket | 72.0 ± 2.8 | 23.9 ± 3.4 | 1356 |

---

## 📝 引用与发表信息

本研究工作目标投稿至 **Sleep Medicine** 期刊，是睡眠医学领域的权威学术期刊。

---

## ⚕️ 伦理声明

本研究使用的所有多导睡眠图数据均来自已公开的三大公共数据库（Sleep-EDF、ISRUC-Sleep、DREAMS），原始数据收集阶段已经获得对应机构伦理审查委员会的批准并获得所有受试者的知情同意。由于本研究使用完全去标识化的公开数据集，因此无需额外IRB伦理审查。

---

## 📧 联系方式

如有任何问题，请通过通讯作者邮箱联系我们。
