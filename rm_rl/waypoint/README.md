# 保守雷达选点模型（带滞回）

当前只保留 `rm_runs/radar27_cls_v1/best.pt` 这一套已训练权重。滞回是权重外的运行逻辑，不是另一个模型文件。

| 项目 | 当前保留版本 |
|---|---|
| 输入 | `radar27-ground-v1`，315 维，含数值、有效位、信息年龄 |
| 网络 | 315 → 256 → 256 → 4 |
| 输出点（红方坐标） | 中央 `(10.8, 7.5)`、A `(11.13, 10.64)`、B `(11.71, 12.08)`、补给 `(1.4, 13.0)` |
| 滞回 | 新目标连续确认 1 秒；旧目标最短保持 2 秒 |
| 紧急补给 | 模型提出补给点，且弹药 <30 或血量比例 <24% 时绕过等待 |
| SHA-256 | `4e9fbb86a0bd4b33a9b60b3ba48dac1ed8e1e2b144d31e2005c5cf796889d9d9` |
| 训练数据 | `data/radar27_pilot_v2/` |
| 历史完整评测 | `rm_runs/radar27_matrix30_v1/`，240 场，含 60 场滞回模型 |

历史滞回模型平均伤害 1567.85、死亡 1.25、队伍胜率 33.3%。这些是旧执行器的结果，不能自动归于修正后的执行器。

## 运行短程验证

从项目根目录执行；Python 需要安装 PyTorch、NumPy，Arena 外部运行时需存在。

```bash
/home/delphine/miniconda3/envs/dl_env/bin/python -m rm_rl.waypoint.radar_sim evaluate \
  --ckpt rm_runs/radar27_cls_v1/best.pt \
  --modes radar_hysteresis --confirm-s 1 --min-hold-s 2 \
  --executor legacy --side red --matches 1 --duration 10 \
  --out /tmp/radar-conservative-smoke
```

`legacy` 保留历史评测的执行时序，其中存在移动/姿态判断先于目标替换的已知局限。修正实现 `goal-before-posture-v2` 仍保留，默认使用修正实现；需要复现历史时显式传 `--executor legacy`。切换执行器后应单独评测，不能混入历史结果。

`radar_sim evaluate` 的默认 checkpoint 已指向保留权重，默认模式为 `radar_hysteresis`。双边配对测试使用 `radar_matrix`，仍须显式指定 checkpoint。`radar` 模式仅关闭同一权重外的滞回，供对照用，并非第二个模型。

## 保留的依赖

- `radar.py`：模拟雷达输入边界、有效位和年龄。
- `model.py`、`radar_sim.Predictor`：加载原权重并选择目标点。
- `goal_filter.py`：滞回；`radar_sim.Candidate`：紧急补给、失效回退及坐标转换。
- `executor.py`：修正后的目标替换时序。
- `teachers.py`、`train_cls.py`、`action_space.py`：保留通用蒸馏能力。
- `radar_matrix.py`：可恢复的配对评测。

前压模型、两版前哨归属估计器及其专用流水线已移出当前项目；103 场实验的统计与原始逐场结果保留在 `rm_runs/reports/opus55_s27o_assault/`。其他早期模型权重和冒烟实验产物也已移出。原始 OfflineRL 通用算法与数据集未进行大规模改造。

后续蒸馏流程见 `docs/RADAR_AGENT_DISTILL_SOP.md`。当前模型仍是仿真研究版本，未完成 ROS 2 消息、串口协议或实机验证。

## 2027 正式协议输入版本（2026-10-29 开始重蒸馏）

旧权重保留。新输入 `radar27-protocol-20261029-v1` 为 106 个字段 × 3 = 318 维，字段、来源命令、类型转换和频率见 `configs/radar/protocol27_v1_contract.json` 和 `protocol27.py`。这是一套有意选取的协议子集，不代表用尽雷达可收到的所有信息。

`protocol27_sim.py` 在传感器边界把 Arena 状态编码为命令载荷，按协议源频率、延迟和丢包投递，再走与真实载荷共用的 `ProtocolInput` 解码/归一化入口。直接广播、雷达无线、哨兵回传分开限制；回传模拟两跳，原始源时间戳不会因转发刷新。自身位置、队友位置及己方姿态使用 1 Hz 源数据。训练/评测不读取解码器之外的战场真值。

本版加入直接报告的阵地控制状态和控制度；己方普通允许发弹量与堡垒储备分开，敌方 3/4/7 允许发弹量包含仿真共享堡垒储备。两个阵营提供的控制状态分别保留，`己方未控制` 不会被推断为 `敌方完全控制`。区域激活暂时省略，不能将 Arena occupants 当协议激活；己方无敌/虚弱、航向、金币等也不作为本版必需输入。强化姿态只保留其基本姿态。

接入 ROS 时仍需完成：真实帧 CRC/命令解析、无线解密或解码来源校验、0301 哨兵发送者身份与封包/带宽检查、统一时钟，以及裁判系统坐标到 Arena 地图的标定。`ProtocolInput` 接受已验证的 payload，不是串口驱动。这里的小端载荷约定需要用实机抓包复核，仿真使用 identity 坐标映射不代表实机场地已标定。回传消息的来源时间必须随自定义消息保留，不能用雷达收到转发的时刻替代。

实验：40 场教师采集（14000–14019，各红蓝），按种子分组训练/验证，四点覆盖审计通过才训练；距所有固定点超过 0.05 m 的样本保留在原始数据中，但不进入四点训练副本。若训练集或验证集的剔除比例超过 0.5%，则中止。40 epoch 后在 14100–14101 做开发闭环，14200–14204 做 40 场四策略评测，再使用旧权重与其旧输入假设补做 10 场参考。共 50 场最终评测，均使用修正执行器。旧/新模型输入假设不同，差值是系统版本比较，不能解释为相同观测下网络结构的优劣。

采集完整结束后运行：

```bash
/home/delphine/miniconda3/envs/dl_env/bin/python scripts/arena/retrain_protocol27.py
```

可用 `--wait-for-collection` 等待正在进行的采集。产物在 `rm_runs/radar27_protocol_v1/`，状态在 `pipeline_status.json`；任何审计或子进程错误都会停止，不会偷偷跳过。已有流水线输出不自动覆盖。

## 普通哨兵学生实验

`python scripts/arena/distill_baseline27.py` 执行普通 `balanced` 哨兵的两点蒸馏，输入保持 318 维新协议合同，输出中央/补给两个坐标，沿用 1 秒确认与 2 秒最短保持的滞回。现有模型保留。产物目录 `rm_runs/radar27_baseline_v1/`，采集目录 `data/radar27_baseline_v1/`。流水线拒绝覆盖已有实验；审计失败停止。50 场最终评测含 10 场现有新协议滞回模型对照，两者输入条件相同。
