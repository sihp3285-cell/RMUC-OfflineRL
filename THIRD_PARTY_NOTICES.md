# Third-Party Notices / 第三方材料声明

本仓库的 [MIT License](LICENSE) **仅覆盖本项目自身的代码与文档**。以下材料不在
其范围内,各自权利归原权利人所有。

## 上游项目

本仓库派生自 [Harkerbest/RMUC-OfflineRL](https://github.com/Harkerbest/RMUC-OfflineRL)
(MIT License, © 2026 Harkerbest)。基础数据管线、161 维观测契约、IQL/BC/DT 算法
实现与可视化工具来自该项目,其原始版权声明随 [LICENSE](LICENSE) 保留。本仓库在
其之上扩展了目标标签、攻击归属辅助监督与评估口径。

## RMUC 2026 区域赛数据集

比赛数据由大疆创新(DJI)/ RoboMaster 组委会官方发布,版权归其所有。本仓库
**不包含、不分发**任何比赛数据,请自行从官方渠道获取(见 [README](README.md)
「快速开始」一节)。

由此派生的产物同样不予分发:`viz/replays/*.json` 是单场比赛逐秒裁判系统状态的
机器可读再编码,等同于数据集切片,已在 `.gitignore` 中排除。仓库内保留的
`viz/figures/*.png` 为聚合统计图表,属衍生分析结果。

## 场地俯视图

`viz/assets/rmuc_2026_field_top_view.jpeg` 出自 RMUC 2026 规则手册,版权归
大疆创新(DJI)所有,**不随本仓库分发**,需使用者自行放置,详见
[viz/assets/README.md](viz/assets/README.md)。

仓库内 `viz/figures/*.png` 中,部分图表以该场地图为底图(经压暗、去饱和处理),
就此范围而言同样不在本仓库 MIT 协议覆盖之内。用
`--no-field-image` 重新生成即可得到不含该素材的版本。

## 场地标定参数

回放器与绘图脚本使用的场地裁切标定(源图 1683 × 938 px,有效场地内框
(100, 69, 1576, 856) px)取自开源项目
[ezthor/rm-battlescope](https://github.com/ezthor/rm-battlescope)
(MIT License),该项目同样基于本届官方数据集。

## 商标

RoboMaster、RMUC 及相关名称与标识为大疆创新(DJI)的商标。本项目为独立的
第三方研究工作,**未获得 DJI 或 RoboMaster 组委会的授权、认可或背书**。
