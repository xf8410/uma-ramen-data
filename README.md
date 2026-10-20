# 拉面杯训练数据仓（uma-ramen-data）

**一句话：data754 五数据集的转换矩阵（5.6 万局 × 754 维）+ GBDT/NN 权重 + 每次 CI 的 txt 运行日志——数据不落云盘、不落 /tmp，全进仓库。**

## 为什么单开这个仓

- 2026-10-20 事故：训练特征 `features.npy`（168MB）+ `meta.jsonl` 放在 /tmp 临时盘，环境重启被清空，找不回来
- 用户纪律：**数据不放云盘工作区，单独公开仓托管；每次 CI 跑完，数据日志自动放进权重仓库；权重跑的局数资料必须是 txt 可读格式**
- 本仓 = 数据仓 + 新权重仓库（接替 umaai-rs `weights` 分支的权重托管职责）

## 目录结构

```
data/            # 转换矩阵（convert754 产出，按数据集分目录）
  prod_n4096/    #   features.npy [N,754] f32 + legal_mask/score_mean/score_n + meta.jsonl + summary.json
  r2roll/        #   23526 样本（主力批）
  r3roll/        #   ~9186 样本
  r4roll/        #   ~1.05 万样本
  r4roll2/       #   ~1.14 万样本
weights/         # 模型权重（gbdt_model.json 等，训练完自动入仓）
logs/            # 每次运行一份 txt：局数/平均分/训练指标/产物清单+md5
```

原始数据（bincode 分片包，1.58GB）在 [xf8410/umaai-rs `data754` 分支](https://github.com/xf8410/umaai-rs/tree/data754)，本仓只存转换后的矩阵与产物。

## CI 重训流水线（retrain.yml）

`workflow_dispatch` 或 push `scripts/**` 触发，全程 GitHub 免费算力（公开仓不限时）：

1. checkout umaai-rs `feat/gbdt-leaf`（convert754 + gbdt_export.py，**不改上游源码**）
2. checkout umaai-rs `data754`（原始分片包）
3. 拼装分片 → **md5 硬校验**（对不上即红）→ 解包
4. `convert754` 逐数据集转换 → `data/<数据集>/`
5. 合并五数据集 → 训 GBDT（200 树 d4 lr0.1，random_state=42 全确定）
6. `weights/gbdt_model.json` + `logs/retrain_<时间戳>.txt` 自动 commit 进本仓

## txt 日志格式（每跑一次一份）

```
logs/retrain_20261020_083000.txt
  【数据】各数据集样本数 + 平均分 + 合计（对照历史 55777）
  【训练】gbdt_export.py 原始输出（MAE / R²）
  【产物】权重+数据文件清单 + md5
  【结论】与历史参考值（MAE≈711 / R²≈0.86）的偏差
```

## 历史参考值（2026-10-20 首训）

| 项 | 值 |
|---|---|
| 合计样本 | 55777（≈5.6 万局） |
| 各数据集平均分 | 63772 ~ 64037 |
| GBDT（200 树 d4） | MAE=711，R²=0.86 |

## 与其它仓的关系

- 数据原始来源：umaai-rs `data754` 分支（不动）
- 训练/推理代码：umaai-rs `feat/gbdt-leaf` 分支（不动）
- 实验结论归档：uma-algo-race / uma-ramen-nn-lab 主账本
- 本仓只收：**数据矩阵 + 权重 + txt 日志**，不收代码
