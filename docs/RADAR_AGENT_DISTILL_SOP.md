# 雷达选点模型蒸馏与滞回评测 SOP

版本：v1，2026-10-27。范围：当前仓库中的 Arena 教师策略 → 雷达观测学生 → 坐标选择 → 滞回 → 仿真哨兵执行。

## 1. 目标与产物

将一个教师 agent 的“选择目标位置”行为，通过监督学习训练成小型神经网络。网络只负责选点；导航、姿态、瞄准、射击由执行器负责。当前学生不是强化学习算法，闭环测试不会更新权重。

最终版本必须是一套文件与配置，而非单独一个 `best.pt`：

- 教师入口、依赖及版本/哈希；
- 数据集、采集 manifest、训练/验证对局清单；
- 输入字段合同、候选坐标表、归一化参数；
- `best.pt`、`metrics.json`、分类诊断；
- 滞回参数、回退策略、执行器版本；
- 独立测试的 manifest、逐场结果、轨迹、配对统计；
- 已验证范围与未解决问题。

流程：冻结合同 → 教师兼容检查 → 小规模采集 → 数据审计 → 正式采集 → 训练 → 离线诊断 → 固定滞回 → 小规模闭环 → 冻结版本 → 独立大样本测试 → 归档。

## 2. 已有最新版本的准确记录

| 项目 | 已有版本 |
|---|---|
| 教师 | `gpt61_2027_adaptive_2772/strategy.py`；Python 策略，不是运行时调用 LLM |
| 学生权重 | `rm_runs/radar27_cls_v1/best.pt` |
| 学习方式 | 硬目标标签的行为克隆；随机初始化学生，不是继承旧 Arena 学生权重 |
| 输入 | `radar27-ground-v1`；105 字段 × 数值/有效性/信息年龄 = 315 个 float32 |
| 网络 | 315 → 256 → 256 → 4；隐藏层 Linear、LayerNorm、ReLU；148,740 参数 |
| 输出 | 四个固定候选点的分数；取 argmax，再查坐标 |
| 数据 | `data/radar27_pilot_v2`；红方 8 场，种子 8100–8107 |
| 训练/验证 | 6 场 23,793 条 / 2 场 8,399 条；验证对局 8100、8105 |
| 采集扰动 | 额外延迟 0、丢包 0、位置噪声 0；仍保留通道采样频率 |
| 训练 | 40 epoch，batch 1024，lr 0.0003，AdamW，weight_decay 0.0001，余弦学习率，类别加权、0.05 标签平滑、梯度裁剪 1.0 |
| 权重选择 | 验证 top1 最大，平手比较坐标误差中位数；保留第 5 epoch |
| 滞回 | 新目标连续 1 秒且旧目标保持至少 2 秒；与原雷达模型共用权重 |
| 独立测试 | 种子 8400–8429 × 红蓝 × 4 策略，240 场；420 秒；延迟 0.13 秒、丢包 0.01、位置噪声 0 |
| 测试目录 | `rm_runs/radar27_matrix30_v1` |

候选点（红方统一坐标，米）：中央 `(10.8,7.5)`、A `(11.13,10.64)`、B `(11.71,12.08)`、补给 `(1.4,13.0)`。蓝方通过 `(28-x,15-y)` 变换。

历史结果：验证 top1 82.3%，恒选中央点为 86.3%；验证集没有 A 类，补给召回约 55.7%。240 场中滞回减少约 83% 切点，但没有确认相对原模型的伤害/生存收益。此前 4 种子中 +18.8% 伤害未复现。不得把这些结果作为所有新教师的验收阈值或保证。

## 3. 开始前必须确认的边界

### 3.1 输入合同

当前合同在 `rm_rl/waypoint/radar.py` 的 `FIELDS`、`contract()` 中。它是暂定语义合同，不是已验证的 2027 串口协议。包含假设的哨兵回传、全场敌方信息及仿真代理字段。

若任务目标是实机部署，必须先建立逐字段来源表：字段、协议/ROS 消息出处、原始类型、单位、坐标系、更新频率、时间戳依据、无效/过期条件。没有来源的字段不能因为仿真可读就加入输入。接收端生成的有效标记与年龄，应与额外战场状态区分。

若仅进行算法研究，可继续使用现合同，但所有结果须标注“暂定雷达输入条件下的仿真结果”。字段、顺序、单位、年龄编码、槽位或坐标语义变化时，升级 schema、重新采集和训练；不能只补零或改维度继续使用旧权重。

### 3.2 教师兼容性

现有采集器 `TracedSentryTeacher` 要求：

- 教师是可导入的策略模块，提供 `act(observation, state)`，返回动作和新状态；
- 教师通过模块级 `drive(robot, goal, obs, memory, tolerance)` 传递哨兵目标；
- 跟踪器只记录 slot 7、role sentry 调用中的目标；每次 act 前清空旧目标；
- 采集时教师控制哨兵，其他己方机器人由普通策略控制；不是原教师控制全队的完整行为蒸馏；
- 教师依赖自己的观测和记忆；学生只有当前雷达向量。必须记录教师独有信息与历史状态可能导致的不可模仿性。

新 agent 不调用这个 drive、内部缓存了函数引用、直接输出速度、一次为哨兵调用多个候选目标，或返回不同接口时，先做专用适配与测试。不能直接给路径并假设标签正确。对多个 drive 调用，当前采集器会保存最后一个匹配目标，要确认它确实是实际执行目标。

更换教师后，先查输出目标是否被四点覆盖。当前雷达训练器始终使用 `radar.ANCHORS`；`--max-anchors` 不会让雷达分支自动学新锚点。所有目标都会被映射到最近锚点，映射误差过大必须处理，不能继续训练后当作完整复制教师。

### 3.3 执行器已知问题

当前 `Candidate.drive` 替换导航目标，但普通哨兵在函数外的部分移动/姿态判断仍引用原策略目标。这一不一致已确认，尚未修复。进行新的可归因性能结论或实机准备前，应修复并测试，建立新版本；历史 240 场保留原样，不覆盖或混算。

教师采集使用教师执行逻辑，学生测试使用普通执行逻辑。因此教师目标相同不保证效果相同。现有 CLI 没有教师评测 mode；需要比较教师原版或“教师只选点＋同一执行器”时，先实现明确的评测适配，不把 baseline 冒充教师。

## 4. 准备一个新实验

以下命令在仓库根目录执行。`teacher_path` 必须换成真实的新教师入口，`run_id` 必须唯一。对同一教师公平比较时，其他实验条件保持一致。不同版本的数据、权重与评测目录不可覆盖。

```bash
cd /home/delphine/rm/RMUC-OfflineRL-main
py=/home/delphine/miniconda3/envs/dl_env/bin/python
run_id=agent_b_v1
teacher_path=/absolute/path/to/new_agent/strategy.py
baseline_path=/home/delphine/rm/arena_strategy_backup/strategy3_baseline.py
opponent_path=/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app/examples/claude_opus55_xhigh_2027_hold/strategy.py
"$py" -m unittest discover -s tests -v
```

保留教师依赖和配置：现有采集 manifest 记录入口文件 SHA，但不会自动递归锁定教师所有外部依赖。新 agent 若依赖其他模块、模型文件或随机状态，另存这些版本/哈希，并固定相应随机种子。

预先约定：训练/验证场景、开发测试种子、最终测试种子、扰动设置、对手、候选点、主指标与可接受退化范围。不要看最终测试结果后回头改通过标准。

## 5. 教师采集与数据审计

### 5.1 短采集检查

```bash
"$py" -m rm_rl.waypoint.radar_sim collect \
  --teacher "$teacher_path" --baseline "$baseline_path" --opponent "$opponent_path" \
  --side red --matches 1 --duration 30 --workers 1 --seed-start 9990 \
  --latency 0.13 --loss 0.01 --noise 0 \
  --out "data/${run_id}_smoke"
```

短测试只验证能运行和生成标签，不能说明教师覆盖了补给/死亡/恢复等场景。检查 teacher drive 的实际目标、标签数、坐标合法性，并留意教师内部异常是否被它自己吞掉；现有采集器没有完整的教师异常计数。

### 5.2 正式采集模板

下面是新实验模板，不是历史 8 场的复现参数。每侧 20 场只是起始规模，不是“数据足够”的保证；根据类别和场景覆盖追加采集。这里主动加入与主要评测一致的输入扰动，最终仍需单独测试其他扰动条件。

```bash
"$py" -m rm_rl.waypoint.radar_sim collect \
  --teacher "$teacher_path" --baseline "$baseline_path" --opponent "$opponent_path" \
  --side red --matches 20 --duration 420 --workers 4 --seed-start 10000 \
  --latency 0.13 --loss 0.01 --noise 0 \
  --out "data/${run_id}_red"

"$py" -m rm_rl.waypoint.radar_sim collect \
  --teacher "$teacher_path" --baseline "$baseline_path" --opponent "$opponent_path" \
  --side blue --matches 20 --duration 420 --workers 4 --seed-start 10100 \
  --latency 0.13 --loss 0.01 --noise 0 \
  --out "data/${run_id}_blue"
```

当前采集器每约 0.1 秒记录雷达特征与同一时刻教师的目标，有有效目标才保存。教师答案是当前决策，输入可能因通信而较旧；不会偷偷给学生补未来信息。

采集后、训练前检查：

1. 每个 shard 的 schema、维度、时间顺序与传感器配置正确，特征/目标没有 NaN 或无穷。
2. 红蓝输入与输出坐标成对验证；不是仅把蓝方目标旋转、忘记输入朝向与位置。
3. 统计四类标签、每场分布、缺失率与信息年龄；训练和验证都要覆盖关键行为。相邻帧高度相关，样本数不等于独立场景数。
4. 计算教师目标到最近候选点的均值、P95、最大误差，并检查超出预先允许误差的比例。阈值应根据场地/导航容差制定；目标不被覆盖时先改输出设计，不接受静默最近点替代。
5. 训练/验证按整场隔离。当前训练器按传入目录顺序收集文件，每第 5 个文件进入验证；检查实际 `train_files` / `val_files`。同一原始对局的扰动副本、重放副本和同 seed 的关联换边数据应放在同一分组；当前自动分割不负责识别这种关联，需要在数据组织或分割代码中显式处理。
6. 当前模板红蓝使用不同采集种子，降低同 seed 关联分割风险；开发测试与最终测试使用全新种子。

不得仅因“训练能启动”而跳过这些检查。

## 6. 训练学生

```bash
"$py" -m rm_rl.waypoint.train_cls \
  --data "data/${run_id}_red" "data/${run_id}_blue" \
  --out "rm_runs/${run_id}_cls" \
  --epochs 40 --batch 1024 --lr 0.0003 --hidden 256 --depth 2 \
  --val-every 5 --seed 0
```

确保输出目录未被使用：训练器不像采集器那样拒绝非空目录，会覆盖同名产物。

训练器从训练集计算归一化参数，将目标映射到候选点，使用加权交叉熵训练。实际批次、损失与权重选择以 `metrics.json` 为准。训练 40 轮不意味着保留第 40 轮。

必须检查：

- 训练/验证文件与样本数；类别计数；目标映射误差；
- 混淆矩阵、每类支持数/召回率，特别是补给类；某类零样本应写“无法评估”，不是证明召回差；
- 与恒选中央点/多数类等简单基准比较，避免多数类制造高准确率；
- 准确率和坐标误差只代表模仿程度，不代表胜率或目标到达率；
- `best.pt` 内的 schema、完整合同、anchors、mean/std、网络维度与预期一致。

当前训练 CLI 自动生成 `metrics.json` 与 `norm_stats.npz`，不会自动生成历史目录中的 `diagnostics.json`；混淆矩阵等需另行计算，不能以为运行上述命令就已经做完诊断。

## 7. 加上固定滞回

不是另训一个网络：`radar` 与 `radar_hysteresis` 使用同一 `--ckpt`。

- 初始目标立即接受。
- 新目标连续提出 ≥1 秒，且当前目标已保持 ≥2 秒，才接受普通切换。
- 新目标变化或返回当前目标时，连续确认被重置。
- 仅当模型提出补给点，且有效信息显示弹量 <30 或血量比例 <24%，才立即放行；不是强制撤退策略。
- 定位缺失或预测失败清空滞回并由普通策略接管；观测到死亡会重置滞回。真实通信超时与哨兵端接管尚未实现。
- 滞回使用仿真时间，不以程序运行耗时计时；当前普通推理不按网络置信度设门槛。

新教师若改变候选表，需要重新检查补给点标识。当前紧急逻辑依赖 `ANCHORS[3]`；不能随意调换类别顺序。

## 8. 先做开发闭环，再冻结版本

红蓝短测试与 4 种子完整开发测试使用不同目录。开发阶段允许调整参数，但不能把这里的数据当最终泛化证据。

```bash
for side in red blue; do
  "$py" -m rm_rl.waypoint.radar_sim evaluate \
    --ckpt "rm_runs/${run_id}_cls/best.pt" \
    --baseline "$baseline_path" --opponent "$opponent_path" \
    --modes baseline rule radar radar_hysteresis \
    --confirm-s 1 --min-hold-s 2 --side "$side" \
    --matches 1 --duration 10 --workers 2 --seed-start 10900 \
    --latency 0.13 --loss 0.01 --noise 0 \
    --out "rm_runs/${run_id}_smoke_${side}"
done

for side in red blue; do
  "$py" -m rm_rl.waypoint.radar_sim evaluate \
    --ckpt "rm_runs/${run_id}_cls/best.pt" \
    --baseline "$baseline_path" --opponent "$opponent_path" \
    --modes baseline rule radar radar_hysteresis \
    --confirm-s 1 --min-hold-s 2 --side "$side" \
    --matches 4 --duration 420 --workers 4 --seed-start 11000 \
    --latency 0.13 --loss 0.01 --noise 0 \
    --out "rm_runs/${run_id}_dev_${side}"
done
```

检查：预测/控制器异常为零，输出合法，回退次数与预期输入缺失一致；查看轨迹中的模型提议和实际目标，检查补给响应、抖动、卡住、死亡恢复。只有这些检查合理后才进行长跑。

## 9. 独立大样本评测

下面模板用种子 12000–12029，红蓝双方各 30 × 4 场，总计 240 场。同一对比实验保持执行器、队友、对手、通信条件、时长一致，只改变待比较策略。

**必须显式指定新 checkpoint。当前 `radar_matrix.py` 内的路径和参数硬编码为历史模型，不可直接用于新 agent。** 以下采用已支持 `--ckpt` 的 `radar_sim`；它逐场写结果，但不支持自动断点续跑，输出目录必须为空。若需要可恢复长跑，先参数化 matrix 的 checkpoint/对手/执行器/滞回与通信选项并验证 manifest 的恢复校验，再启动新实验。不要修改正在运行的实验代码或伪改旧 manifest。

```bash
for side in red blue; do
  "$py" -m rm_rl.waypoint.radar_sim evaluate \
    --ckpt "rm_runs/${run_id}_cls/best.pt" \
    --baseline "$baseline_path" --opponent "$opponent_path" \
    --modes baseline rule radar radar_hysteresis \
    --confirm-s 1 --min-hold-s 2 --side "$side" \
    --matches 30 --duration 420 --workers 6 --seed-start 12000 \
    --latency 0.13 --loss 0.01 --noise 0 \
    --out "rm_runs/${run_id}_test_${side}"
done
```

同一组候选教师的公平评比可共用预先封存的测试种子；一旦根据这些测试结果修改模型，它们就成为开发数据，下一轮最终结论要用新的留出集合。

汇总两侧（调用现有聚合函数，按 seed 合并红蓝后 bootstrap，不把 60 场当成 60 个独立 seed）：

```bash
"$py" - "$run_id" <<'PY'
import json, sys
from pathlib import Path
from rm_rl.waypoint.radar_matrix import aggregate
run_id = sys.argv[1]
rows = []
for side in ('red', 'blue'):
    p = Path('rm_runs') / f'{run_id}_test_{side}'
    rows.extend(json.loads((p / 'results.json').read_text()))
expected = {(seed, side, mode) for seed in range(12000, 12030)
            for side in ('red', 'blue')
            for mode in ('baseline', 'rule', 'radar', 'radar_hysteresis')}
actual = {(r['seed'], r['side'], r['mode']) for r in rows}
assert actual == expected and len(rows) == len(expected), '不完整或存在重复场次'
assert all(r['duration_s'] == 420 for r in rows), '存在非完整时长对局，需单独审查'
assert len({r['engine_hash'] for r in rows}) == 1
assert len({r['rules_hash'] for r in rows}) == 1
out = Path('rm_runs') / f'{run_id}_test_summary.json'
assert not out.exists(), '汇总已存在，先检查版本，不覆盖'
out.write_text(json.dumps(aggregate(rows), ensure_ascii=False, indent=2))
print(out)
PY
```

该汇总检查身份/数量/引擎一致性；还需检查两侧 manifest 的 checkpoint、执行器、对手及传感器配置一致，确认不是拿了别的实验目录。若合法规则允许提前结束，对时长断言的失败应审查而非强行补齐时间。

## 10. 如何判读与决策

主指标在开跑前指定。历史实验以造成伤害为主，死亡、存活、胜率、切点等为辅助；未来若以守点为主，应先在仿真中实现相应指标，不得把伤害默认当作最终任务目标。

- 展示红蓝分别结果与合并结果，并报告按 seed 配对的差值和置信区间。
- 滞回与原学生共用权重与输入，主要用于判断输出稳定机制的影响。
- 普通哨兵使用原 Arena 观测，其比较是系统级比较；规则与学生共用雷达边界。
- 均值更好但置信区间跨零，写“收益未确认”；不能写“效果相同”或“已证明不劣”。非劣性需要预先指定允许差值。
- 到达率分母是实际目标段，减少切点会改变分母。联合查看目标附近停留时间、到达段数、成功段耗时、伤害、生存，不把到达率单独当导航进步。
- 成功段到达耗时不包含失败段；`stuck_cmd_s` 是距离目标 >1m 且速度 <0.1m/s 的代理，不是经过验证的所有卡死事件。
- 净 HP 损失只是受伤代理，可能漏掉同一时间的治疗/伤害抵消。
- 当前 matrix 聚合不提供胜率差置信区间，需要额外按 seed 计算，不能把胜率点估计直接解释为统计优势。
- 当前固定对手与固定扰动不能证明所有局势泛化。后续追加不同对手、延迟/丢包、位置误差、关闭回传/敌方通道等独立评测；先确认哪些字段本来就能获取。

验收结论应选明确层级：管线已跑通 / 学会部分教师行为 / 指定仿真条件下有收益 / ROS 消息回放一致 / 实机链路验证通过。当前历史版本只达到仿真研究范围，未通过真实协议、ROS 2 回放或实机部署验收。

## 11. 下次更换 agent 的最短检查单

- [ ] 教师路径、依赖、观测条件、控制对象与目标提取方式已验证。
- [ ] 输入有来源；暂定字段明确标注；schema 和坐标合同冻结。
- [ ] 新教师目标被候选点覆盖，或输出设计已经升级。
- [ ] 已处理执行目标与底层移动/姿态判断不一致问题；不混入旧版结果。
- [ ] 数据覆盖红蓝、补给、守点、死亡恢复和关键转移，整场分割且无重放泄漏。
- [ ] 分类诊断完成，不仅看总体准确率。
- [ ] 原学生和滞回版引用同一新权重；补给类别标识正确。
- [ ] 开发测试通过后冻结参数，最终测试无调参污染。
- [ ] 新权重被显式传给评测器，没有误跑硬编码旧模型。
- [ ] 逐场轨迹、完整版本记录、配对统计与局限已归档。

历史相关规划文档中的协议速率、字段归属或验收阈值，不应自动视为已核实事实。实际实现以代码与各实验 manifest 为准；真实来源以正式协议及 radar27 的解析/消息发布实现核对。
