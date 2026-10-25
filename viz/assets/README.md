# 场地底图(需自行放置)

回放器和图表可以用 RMUC 2026 场地俯视图作为底图。**这张图不随本仓库分发** ——
它出自 RMUC 2026 规则手册,版权属大疆创新(DJI),不在本项目 MIT 协议范围内,
详见仓库根目录的 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。

把文件放成:

```
viz/assets/rmuc_2026_field_top_view.jpeg
```

放好后回放器会自动启用(页面上的「场地图」按钮可随时开关),
`heatmaps.py` / `policy_field.py` 加 `--field-image` 即可。没有这个文件时,
所有工具照常工作,只是底图退化成纯网格。

## 从哪拿

- **RMUC 2026 规则手册**里的场地俯视渲染图,自行裁切导出。
- 或者取自同样基于本届数据集的开源项目
  [ezthor/rm-battlescope](https://github.com/ezthor/rm-battlescope)
  的 `assets/rmuc_2026_field_top_view.jpeg`(该项目代码为 MIT,
  但这张图同样被排除在其 MIT 范围之外)。

```bash
curl -L -o viz/assets/rmuc_2026_field_top_view.jpeg \
  https://raw.githubusercontent.com/ezthor/rm-battlescope/main/assets/rmuc_2026_field_top_view.jpeg
```

## 标定

裁判系统的坐标原点**不是**这张图的角点:追踪坐标从有效区与停机坪的内角开始,
所以图上一圈护栏在 (0,0)–(28,15) 之外,必须先裁掉再拉伸到场地框。
差这一步,全场机器人会整体偏移约一米——看得出来,但很容易忽略。

标定沿用 rm-battlescope 的 `rmuc_trajectory/field.py`:

| 项 | 值 |
|---|---|
| 源图尺寸 | 1683 × 938 px |
| 有效场地内框 | (100, 69, 1576, 856) px |
| 对应物理范围 | (0, 0) – (28, 15) m |
| y 轴 | 图像行号向下增大,场地 y 向上增大(需翻转) |

裁切框在代码里以**比例**存储([`viz/field_canvas.py`](../field_canvas.py)
与 `replay.html` 各存一份),所以换成缩放过或重新导出的图也能对齐。
