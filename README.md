<div align="center">

# RMUC-OfflineRL

**基于公开赛事日志的 RoboMaster 全兵种自主决策离线强化学习框架**

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-research-yellow)

</div>

---

## 当前保留的雷达决策模型

当前唯一保留的训练权重是 **`rm_runs/radar27_cls_v1/best.pt`**，使用 315 维雷达输入、四个候选点和“确认 1 秒 / 最短保持 2 秒”的滞回逻辑。运行说明见 [保守滞回模型](rm_rl/waypoint/README.md)，蒸馏步骤见 [SOP](docs/RADAR_AGENT_DISTILL_SOP.md)。

前压实验及早期模型产物已移出项目；历史统计仍保留。下文是原 OfflineRL 通用框架与历史研究介绍，其中提及的旧权重路径不代表当前仍保留相应文件。

## 这是什么

RoboMaster 超级对抗赛（RMUC）官方公开了 2026 区域赛的裁判系统日志：613 局真实比赛、约 400 万条逐秒状态。本项目在这份数据上搭一条完整的离线强化学习链路——**日志解析 → 观测/动作/奖励构建 → 多卡训练 → 离线评估 → 部署接口**，并把决策降频到 1 Hz 的战术层：**去哪、打谁、能不能开火**。

核心问题不是"能不能训起来"，而是**监督信号从哪来**。日志里没有动作标注，只有状态。本项目的工作集中在一处：把目标选择（打谁）的标签，从"由观测反推的几何规则"换成"实际被打中的是谁"这样的结果量。

## 来源与边界

- 上游框架派生自 [Harkerbest/RMUC-OfflineRL](https://github.com/Harkerbest/RMUC-OfflineRL)（MIT）。基础管线、161 维观测契约、IQL/BC/DT 三种算法实现与可视化工具来自该项目；本仓库在其之上做了目标标签、辅助监督与评估口径的扩展，并在提交历史中保留完整来源。
- 比赛数据版权归大疆创新（DJI）/ RoboMaster 组委会所有，**不随本仓库分发**，需自行从官方渠道获取。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 本项目为独立的第三方研究，**未获得 DJI 或 RoboMaster 组委会的授权、认可或背书**。

## 数据

| 表 | 行数 | 内容 |
|---|---|---|
| `matches` | 613 | 赛区、对阵、胜方、时长 |
| `timeseries` | 4,015,383 | 某局某秒某机器人的完整状态 |
| `events` | 1,474,178 | 发弹 / 受击 / 能量机关 / 飞镖 / 增益 |

采样率 1 Hz、单局约 420 s、场地 28 × 15 m；`robot_id` 红方 1 英雄 / 2 工程 / 3·4 步兵 / 6 空中 / 7 哨兵 / 10 基地 / 11 前哨站，蓝方 +100。

做任何建模前必须先接受这几条数据事实：

- **没有视线/可见性字段**。坐标来自场地全局定位，不是机器人感知；场地障碍图也不在数据里。
- **`events.目标robot_id` 对发弹与受击几乎全为 NULL**，无法做逐机器人伤害归因 → 奖励只能做到队伍级。
- **`枪口朝向` 是零偏移角度**，wrap 到 [−140, 220]；**−140.0 是"无云台数据"哨兵值**，工程/基地/前哨站恒为此值，瞄准推断前必须排除。
- 升级等级可观测（`最大血量` / `小·大热量上限` / `底盘功率`），这是判断"对面是满级英雄还是一级步兵"的唯一依据。

## 问题建模

**观测 161 维**

| 块 | 维度 | 内容 |
|---|---|---|
| ego | 15 | 位置、高度、血量比、存活、易伤、热量比与余量、功率、朝向 sin/cos、弹量、三个能力档位 |
| 时间 | 2 | 已用 / 剩余 |
| 己方槽位 | 6 × 9 | 相对位置、距离、血量比、存活、能力档位；固定 6 槽含自身 |
| 敌方 | 6 × 12 | 相对位置、距离、血量比、存活、易伤、位置已知、能力档位、交火可见性先验 |
| 建筑 | 8 | 双方基地 / 前哨站血量与存活 |
| 经济 | 4 | 己方剩余/总额、对方总额、差额 |
| 对手先验 | 6 | 双方各 3 个匿名强度标量（严格因果，只用此前比赛） |

蓝方经 180° 镜像到红方坐标系，策略与侧别无关，数据翻倍。

**动作 10 维，三个通道**

| 通道 | 维度 | 分布 | 来源 |
|---|---|---|---|
| 导航子目标 | 2 | Gaussian | 5 s 后位移，米 |
| 交战门控 | 1 | Bernoulli | `累计17mm发弹` 差分，精确标签 |
| 目标选择 | 7 | Categorical 软标签 | 6 台敌方 + 无目标 |

数据里没有可见性字段，用**经验交火图**当代理：统计全赛季"从格子 A 向格子 B 实际开火"的次数占比，2 m 栅格、Laplace 平滑、截断到 [0, 4]。

## 方法：三处针对目标选择的改动

### 1. 为什么用人类步兵数据训哨兵策略

离线 RL 的性能上限是示范者。日志里的哨兵轨迹多半就是各队自己写的行为树；步兵则是人类操作且与哨兵**构造等价**（同为 17mm、满级后同为 400 血 / 260 热上限）。全场统计：

| 指标 | 哨兵 | 步兵3 / 步兵4 |
|---|---|---|
| 走过格数（1 m 栅格，共 420） | **25.4** | 73.2 / 74.2 |
| 位置熵（nats） | **1.76** | 3.49 / 3.50 |
| 前 5 格时间集中度 | **81.4%** | 40.5% / 40.3% |
| 跨队走位余弦相似度 | 0.200 | 0.383 / 0.408 |

哨兵覆盖面积只有步兵三分之一、八成时间钉在 5 个格子里。默认训练目标因此是步兵（步兵3 + 步兵4 合并）。

跨兵种迁移有两个必要条件：队伍块固定 6 槽（含 ego 自身，相对位置为 0），以及**不加兵种 one-hot**（实测加入后，迁移到哨兵会退化成每帧输出完全相同的指令）。

### 2. 目标标签：锥角校准 + 排除空中目标

原始几何标签用 `枪口朝向` 与敌方方位角夹角，取 15° 半角锥、`softmax(−误差/τ)`，τ=8°。问题是 15° 太窄：枪口有滞后、目标在动，真实受害者经常落在锥外，标签器**根本给不到它概率质量**。扫描 7 个锥角 × 5 个温度后取 25°：

| | cone 15° | cone 25° |
|---|---:|---:|
| 标签天花板（受害者拿到最大质量的比例） | 70.50% | **72.88%** |
| 漏掉命中（受害者在锥外） | 7.58% | **1.88%** |

空中机器人是无人机，不是合法地面目标，但几何标签会给它质量（锥内只有它时更是整份质量）。处理方式是三处一致：标签里把它清零并重归一化（锥内只剩空中则记为"无目标"）、网络里对它的 logits 置 −1e9、部署解码时先清零再归一化。动作宽度仍是 10 维，**旧 checkpoint 照常加载**。

### 3. 攻击归属作为辅助监督（`aux victim`）

这是本仓库最关键的一处。几何标签是观测的**闭式函数**——它由观测里的 `sin/cos(朝向)` 和敌方相对位置算出来，因此网络只要学会当标签器就能把准确率刷得很高，却不代表战术正确。

实际被打中的是谁是**结果量**，同一观测下并不唯一确定。用 [ezthor/rm-battlescope](https://github.com/ezthor/rm-battlescope)（MIT）从同一份日志推断出 613 局、107,271 条攻击归属，取其中 high-confidence、非歧义、地面机动受害者，得到每局每秒的"实际受害者"，作为辅助标签：

```
L = L_IQL + λ · CE(target_logits, victim)      # 仅命中秒生效，λ = 0.1
```

覆盖率只有训练转移的 1.59%，但信号落在几何标签失效的地方——**锥内有多个敌方**的歧义秒。

## 结果

验证集：61 局比赛、239 条单机轨迹、99,570 条转移；训练/验证比赛 ID 无交集。评估对象是 `best.pt`。

**目标选择（held-out，与攻击归属一致率）**

| 版本 | 一致率 | 标签天花板 |
|---|---:|---:|
| 原始 v1（cone15，无 mask） | 54.31% | 70.50% |
| cone25 | 60.25% | 72.88% |
| cone25 + aux victim | 69.44% | 72.88% |
| cone25 + aux + 推理时 mask | 72.12% | — |
| **cone25 + aux + noair 重训** | **72.88%** | **77.12%** |

同一批 1,600 个 high-conf 命中秒上比较；按整局比赛做 10,000 次配对 bootstrap，`noair` 相对原始 v1 的提升为 **+18.56 个百分点**（95% 区间 [+15.65, +21.61]），相对"v1 + 同一套 mask"为 **+15.13**（[+12.55, +17.91]）。增益集中在多锥样本：45.06% → **60.59%**，单锥 85.03% → 88.01%。

**其余两个通道基本持平**（全量验证转移统一聚合）：`nav_mse` 0.0563 → 0.0565，`fire_f1` 0.4340 → 0.4313。

## 怎么读这些数字

1. **72.88% 是对启发式归属标签的一致率**，不是实战命中率，更不是胜率。归属本身的 high-confidence 里只有约一半与几何标签一致，代理真值带噪。
2. **同批样本上几何规则直接选是 77.12%，仍高于模型**。模型还没超过一行规则，只是在逼近它。
3. **没有胜率证据**。胜负对齐指标在导航/开火/目标三个通道上都没有一致的正向 excess；离线手段在 420 步时域上判定"两个行为不同的策略谁更强"是不可行的，这需要闭环对战。
4. **单因素归因不成立**：不同版本的数据奖励权重并不完全一致，锥角校准也未排除验证集，这些分数应视为开发验证结果。
5. **183 维扩展是负结果**：追加 22 维（复活/买活/增益/堡垒，其中 10 维占位）后归属一致率降至 51.44%。目标头的瓶颈是标签质量，不是状态丰富度。该分支完整保留在 `configs/*_v2.yaml` 与 `scripts/run_v2.sh`，供复核。

一句话：**提升全部来自"把标签从观测的闭式函数换成结果锚定的量"，不来自网络结构、算法选择或特征工程。**

## 快速开始

```bash
# 1) 数据：从官方论坛帖下载后放入 dataset/
#    dataset/rmuc_2026_region_dataset.sqlite

# 2) 环境
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3) 一条龙：交火图 → 队伍先验 → 建数据 → 自检 → IQL/BC/DT → OPE → 导出
bash scripts/run_all.sh
NPROC=4 bash scripts/run_all.sh          # 指定卡数
LIMIT_GAMES=40 bash scripts/run_all.sh   # 冒烟测试
AGENT=sentry DATA=data/sentry_tactical bash scripts/run_all.sh   # 换兵种
```

本轮扩展的完整复现链：

```bash
# 归属证据表（需要 ../rm-battlescope 检出，脚本会记录其 commit）
python scripts/build_target_evidence.py --db dataset/rmuc_2026_region_dataset.sqlite \
    --bs-root ../rm-battlescope --out data/target_evidence.csv --workers 8

# 锥角 / 温度校准
python scripts/calibrate_target_label.py --db dataset/rmuc_2026_region_dataset.sqlite \
    --evidence data/target_evidence.csv --out data/target_label_calibration.csv

# 建数据（cone25 + aux victim + 空中排除）
python -m rm_rl.data.build_dataset --db dataset/rmuc_2026_region_dataset.sqlite \
    --out data/infantry_tactical_cone25_aux_noair --agent infantry --action-mode tactical \
    --target-cone 25 --target-tau 8 --target-evidence data/target_evidence.csv \
    --vis-map data/vis_map.npz --team-prior data/team_prior.json

# 训练与评估
python -m rm_rl.train.train_offline \
    --config configs/infantry_iql_tactical_cone25_aux_noair.yaml
python scripts/eval_target_attr.py \
    --data data/infantry_tactical_cone25_aux_noair \
    --run rm_runs/infantry_iql_tactical_cone25_aux_noair \
    --evidence data/target_evidence.csv --split val
```

训练目录里已有 `best.pt` / `final.pt` 时会拒绝写入，需显式 `--overwrite`——避免一次手滑覆盖掉跑了几小时的结果。**训练产物请用 `best.pt`，不要用 `final.pt`。**

## 代码结构

```
rm_rl/
├── data/
│   ├── schema.py            # 列名 / robot_id / 场地常量 / 朝向哨兵值
│   ├── features.py          # 观测(161) + 动作 + 镜像 + 目标软标签 + 交战门控
│   ├── vis_map.py           # 经验交火图（可见性代理）
│   ├── team_prior.py        # 防泄漏历史队伍强度
│   ├── reward.py            # 结果导向奖励塑形
│   ├── reward_model.py      # 从胜负反演奖励权重（验证用）
│   ├── build_dataset.py     # SQLite → 轨迹 → 转移 → 分片（含 aux victim 标签）
│   ├── obs_v2.py            # 183 维扩展（负结果，保留供复核）
│   ├── check_dataset.py     # 训练前自检
│   └── dataset.py           # 转移集 / 序列集 + 归一化
├── algos/                   # action_spec · networks(HybridPolicy) · iql · bc · dt
├── eval/                    # ope_fqe · ope_dt · win_alignment
├── train/                   # common(DDP) · train_offline · train_dt
├── navigation/              # 导航专用线（557 维观测 / 2 维动作，WIP）
└── deploy.py                # 载入 + 反归一化 + 安全约束

scripts/
├── run_all.sh               # 主流程（各阶段产物一致时自动跳过）
├── run_v2.sh                # obs-v2 独立管线（不触碰 v1 产物）
├── build_target_evidence.py # 攻击归属证据表
├── calibrate_target_label.py# 锥角 / 温度校准
├── eval_target_attr.py      # target_top1_attr（对实际受害者）
├── export_policy.py         # 导出可部署策略
└── collect_results.sh       # 结果打包

viz/                         # 回放器 · 占位分析图 · 策略向量场
jev_demo/                    # 外部模型的对照评估（含其报告）
```

## 部署

```python
from rm_rl.deploy import MLPPolicyRunner, apply_safety

runner = MLPPolicyRunner("rm_runs/infantry_iql_tactical_cone25_aux_noair",
                         device="cuda", camp="红")
cmd = runner.step(obs_161)
# -> {'goal_dx','goal_dy','fire','target','target_conf','target_probs'}

cmd = apply_safety(cmd, heat=h, heat_limit=hl, ammo_left=a, ego_xy=(x, y))
```

`load_policy` 优先加载 `best.pt`；`apply_safety` 施加硬约束（单步位移上限、热量无余量撤销交战许可、弹药耗尽停火、越界抵消向外分量）；蓝方部署自动反镜像。`decode_action` 会把空中目标的概率清零再归一化，因此**旧权重部署时也不会给出非法目标**。

## 可视化

```bash
python -m viz.export_replay --db dataset/rmuc_2026_region_dataset.sqlite \
    --game-id <id> --run rm_runs/infantry_iql_tactical_cone25_aux_noair
python -m viz.serve            # 浏览器打开回放，叠加模型逐秒决策与实际命中
python -m viz.heatmaps         # 哨兵 vs 步兵的占位/离散度/跨队相似度
python -m viz.policy_field     # 冻结某一秒，遍历全场查一次策略（含 --report）
```

场地俯视图属 DJI 素材，不随仓库分发；没有它时所有工具退化为纯网格，功能不受影响。

## 计划

离线阶段已到上限，且离线评估无法判定是否超越现役方案。后续优先级：

1. **闭环仿真**：最小 2D 简化 RMUC 或对接现有模拟器，在线 RL / 自博弈，拿到可信胜率。
2. **实车部署**：对实际不可观测的数据做遮罩后再训练。
3. **数据与算法**：多模态策略（高斯混合 / 扩散）、其余兵种打通、多智能体协同、目标标签的 cone/tau 敏感性分析、半可观测的系统性实验。

## 许可

代码与文档以 [MIT License](LICENSE) 发布。数据、场地素材、第三方项目与商标的归属与使用限制见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 参考

- Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning*, 2021.
- Chen et al., *Decision Transformer: Reinforcement Learning via Sequence Modeling*, 2021.
- RoboMaster 2026 大学联盟赛 / 超级对抗赛规则手册.
