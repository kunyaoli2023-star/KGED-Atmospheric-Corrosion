# V4 数学族、文献来源与稳定性审计

## 审计口径

本报告只读取已有文件，不生成新模块、不重新拟合。模块归族由公开的逐 ID 规则完成；文献提取项的文本分类仅用于排队复核，不能替代原文页码核验。V2 与 V4 的响应变量、单位和数据不同，V2 的数值稳定性不可转移给 V4。

- V4 候选模块：52 条，归入 12 个非空主族；其中 38 条在模块记录中有来源等级，14 条后期模板未记录模块级来源等级。
- 历史文献提取项：334 条，来自 48 篇；全部仍标注为未核验。页面核验主张 10 条，涉及 7 篇。
- V4 第一至六轮累计有 461 个不同结构进入 23 折复核；第一轮 288 个结构全枚举，第二轮以后为筛选后复核或指定对照。族内通过率和中位误差不代表整个扩展候选族。

## 核心数学族与构造依据

| 主族 | 过程问题 | 模块条目 | 测量域内形状或边界 | 当前证据解释 |
|---|---|---:|---|---|
| `T_ACTIVATED` | 温度激活因子 | 1 | 正参数时在 0–50°C 单调上升；温度外推未检验 | A22_EQ4 原式为温度加速因子；迁移到目标总电流需假设 |
| `T_LOCAL` | 温度局部近似或反证 | 3 | 线性或二次局部响应；仅限测量温区 | Arrhenius 局部近似与反证对照；曲率不是独立原文定律 |
| `C_MONOTONE` | 盐的单调促进、抑制或饱和 | 4 | 单调、对数增长或有理饱和；0 浓度需保持有限 | A22_EQ6 为大气氯沉积加速因子；NaCl wt% 与总电流迁移未证实 |
| `C_COMPETING_PEAK` | 盐促进与高盐抑制竞争 | 9 | 促进与线性或二次抑制竞争；内部峰由参数条件限定 | 促进过程有线索；峰形、高盐线性或二次衰减为竞争闭合式 |
| `TC_PEAK_SHIFT` | 温度改变盐峰位置 | 1 | 盐峰位置随温度移动；测量域外形状未约束 | 测量条件下的可反驳交互假设，无原文直接目标函数 |
| `H_MONOTONE` | 液膜可用性或传输单调项 | 2 | 润湿可用性上升或传输随厚度下降；干面极限不外推 | A33_P4 局部阴极电流支持传输过程；总电流及 1/h 需新增假设 |
| `H_COMPETING_PEAK` | 液膜可用性与传输竞争峰 | 6 | 内部峰；对称/非对称与厚膜代数/指数尾互为对照 | 润湿和传输竞争有过程线索；对称性与具体尾部多为描述性形式 |
| `CH_LENGTH` | 盐改变液膜特征长度 | 2 | 盐改变正的特征长度；分母定义域需满足 | 过程启发的盐依赖特征长度，函数形式未由原文证明 |
| `CH_SLOPE` | 盐改变液膜斜率或峰位 | 7 | 盐改变膜厚斜率/峰位；可线性或饱和 | 基于交互诊断的峰位/斜率竞争结构 |
| `CH_PEAK_SHAPE` | 盐改变液膜峰幅度或曲率 | 8 | 盐改变峰曲率/幅度；部分形式强制高盐零曲率 | 基于交互诊断的幅度/曲率结构；高盐零曲率是所选模型边界假设 |
| `TH_COUPLING` | 温度改变液膜响应 | 4 | 温度改变膜厚幅度、峰位或曲率 | 温度–液膜交互为待检验过程假设 |
| `TCH_COUPLING` | 温度和盐共同改变液膜响应 | 1 | 温度与盐共同改变膜厚斜率 | 已有两个交互的组合对照，不是新基本函数族 |

同一族内保留数学上可区分的子型，例如盐峰的线性/二次尾与可变/固定尺度、液膜峰的对称/非对称、高盐端曲率条件。条目数不是独立理论数。

## 原文主张与数学族桥接

下表仅列已页面核验且有模块链接的主张。`PAGE_VERIFIED` 核验的是原文中的主张，并非目标总电流定律。完整 10 条见 `evidence_family_bridges.json`。

| 原文主张 | 原因变量 | 候选主族 | 需要增加的迁移假设 |
|---|---|---|---|
| `A22_EQ4` | temperature acceleration factor for atmospheric corrosion, not measured current | `T_ACTIVATED`, `T_LOCAL` | Does not establish that the target experiment's corrosion current obeys the same factor at fixed NaCl and liquid-film thickness. |
| `A22_EQ6` | acceleration factor indexed by atmospheric chloride deposition | `C_COMPETING_PEAK`, `C_MONOTONE`, `TC_PEAK_SHIFT` | Does not establish response to NaCl wt%, a finite value at zero concentration, saturation, or inhibition. |
| `A11_EQ3` | monthly steel mass loss, g/m^2, at Havana station | `C_MONOTONE` | Does not give a log-current law in solution NaCl concentration or prove global monotonicity. |
| `A10_P3` | reported corrosion rate under evaporating initial moisture layers | `CH_LENGTH`, `CH_PEAK_SHAPE`, `CH_SLOPE`, `H_COMPETING_PEAK`, `H_MONOTONE`, `TCH_COUPLING` | No exact h-response function or salt-film coupling function is given on this page. |
| `A33_P4` | local cathodic current density of Q235 steel in a model electrolyte film | `CH_LENGTH`, `CH_PEAK_SHAPE`, `CH_SLOPE`, `H_COMPETING_PEAK`, `H_MONOTONE` | No published relationship i_lim(h), no whole-cell corrosion-current response to film thickness, and no NaCl-film interaction law are supplied here. |
| `A22_P5_FILM` | atmospheric corrosion under changing wetness | `TCH_COUPLING`, `TH_COUPLING` | This dynamic wetting argument does not by itself establish a temperature-by-thickness interaction when thickness is experimentally fixed. |

## 已验证的结构关系

- `H4_linear_slope_rescaling`：在列明的参数替换或尺度变换后，符号化简严格相等。相关条目不能作为独立族的重复证据。
- `H4_curvature_reparameterization`：在列明的参数替换或尺度变换后，符号化简严格相等。相关条目不能作为独立族的重复证据。
- `H2_fixed_scale_is_free_scale_slice`：在列明的参数替换或尺度变换后，符号化简严格相等。相关条目不能作为独立族的重复证据。
- `H2_bounded_scale_is_free_scale_slice`：在列明的参数替换或尺度变换后，符号化简严格相等。相关条目不能作为独立族的重复证据。

## 历史文献提取项的审计队列

原始分流：CHEMICAL_REACTION 57, PLACEHOLDER_OR_UNSPECIFIED 21, POTENTIAL_RESPONSE_LAW_REVIEW_REQUIRED 69, UNKNOWN_REVIEW_REQUIRED 187。334 条记录中包含时间暴露律、质量损失换算、电化学局部电流、反应式和占位表达，不能统称为 334 条目标腐蚀电流方程。

文本初分流（全部未核验，不能直接引用为原文公式）：

| 文本骨架 | 条目 |
|---|---:|
| `POWER_OR_POLYNOMIAL` | 63 |
| `ADDITIVE_REGRESSION_OR_BALANCE` | 60 |
| `REACTION_OR_STOICHIOMETRY` | 57 |
| `RATIONAL_OR_RATIO` | 54 |
| `EXPONENTIAL_OR_BELL` | 33 |
| `UNSPECIFIED_OR_PLACEHOLDER` | 25 |
| `OTHER_UNRESOLVED` | 13 |
| `LOGARITHMIC_OR_POWER_TRANSFORM` | 12 |
| `RECIPROCAL_TEMPERATURE_OR_ARRHENIUS` | 6 |
| `MEASUREMENT_OR_RATE_CONVERSION` | 5 |
| `PIECEWISE_OR_REGIME` | 3 |
| `ELECTROCHEMICAL_LIMITING_CURRENT` | 3 |

已核验 A22_EQ4、A22_EQ6、A11_EQ3、A33_P4 等主张的来源页，但它们的原目标分别涉及加速因子、质量损失或局部阴极电流；对 69 条潜在响应律仍需按变量、时间尺度和 PDF 页码复核。其余记录保留为线索，不能仅凭自动标签扩充总电流函数库。

## V2 与 V4 的稳定性证据

V2 的 10 次重复生成中，未固定模板的结构签名平均 Jaccard 为 0.2142，canonical floor 为 0.2832；后者核心模板出现率为 1.0，因这些模板预置，说明覆盖稳定而非独立生成成功。其 `family_mix` 全部为 `unlabeled`。

V2 的七个固定核心模板可按拓扑映射至 V4 的温度激活、局部温度、盐峰和液膜峰主族；逐条关系见 `v2_core_crosswalk.json`。此映射只说明数学动机相似，不等于相同因变量、相同浓度/膜厚单位或相同拟合稳定性。

| V4 轮次 | 最低可接受 23 折 log-RMSE | 参数 | 主族组合 | 膜峰采样匹配 | 盐×膜交互绝对残差 |
|---:|---:|---:|---|---:|---:|
| 3 | 0.5574 | 7 | `T_ACTIVATED/C_COMPETING_PEAK/H_COMPETING_PEAK/NULL` | 0.167 | 2.236 |
| 4 | 0.4880 | 8 | `T_ACTIVATED/C_COMPETING_PEAK/H_COMPETING_PEAK/CH_SLOPE` | 0.333 | 1.361 |
| 5 | 0.4713 | 7 | `T_ACTIVATED/C_COMPETING_PEAK/NULL/CH_PEAK_SHAPE` | 0.333 | 0.623 |
| 6 | 0.4519 | 7 | `T_ACTIVATED/C_COMPETING_PEAK/NULL/CH_PEAK_SHAPE` | 0.333 | 0.624 |

### 可以称为核心的层级

| 层级 | 现有证据 | 科学措辞 |
|---|---|---|
| 温度激活主族 | R3–R6 最低可接受模型均含 Arrhenius 形式；A22_EQ4 原式已核验但目标是加速因子 | 当前开发搜索中的稳定候选主干，未证明总电流严格遵循 Arrhenius 或优于所有局部形式 |
| 盐促进–抑制峰主族 | R3–R6 均用有理促进加二次尾的子族；内部盐峰得到测量条件形状支持 | 竞争峰方向较稳；0.05 的尺度和二次尾均为经验结构，不是文献常数或唯一函数 |
| 液膜响应及盐耦合 | R3–R4 为独立液膜峰，R5–R6 为盐依赖联合峰；最终膜峰只匹配 8/24 | 需要环境依赖膜厚响应，但具体对称性、峰位和耦合结构仍不稳定 |

已复核结构的族级汇总如下。一个完整方程同时属于多个族；族的最低误差属于含该族的完整模型，不能当作该族自身的边际贡献。第一轮全枚举，第二轮以后各短名单的选择规则不同，检查数与通过数不应当当作族成功率。

| 主族 | 23 折复核的不同方程 | 数值可接受 | 含该族最低 log-RMSE | 该方程参数 | 雅可比条件数 |
|---|---:|---:|---:|---:|---:|
| `T_ACTIVATED` | 155 | 64 | 0.4519 | 7 | 3.39 |
| `T_LOCAL` | 221 | 23 | 0.4827 | 7 | 3.4 |
| `C_MONOTONE` | 195 | 0 | — | — | — |
| `C_COMPETING_PEAK` | 205 | 81 | 0.4519 | 7 | 3.39 |
| `TC_PEAK_SHIFT` | 6 | 2 | 0.5732 | 9 | 18.6 |
| `H_MONOTONE` | 163 | 22 | 0.5659 | 5 | 2.28 |
| `H_COMPETING_PEAK` | 157 | 44 | 0.4880 | 8 | 5.15 |
| `CH_LENGTH` | 113 | 1 | 0.5732 | 9 | 18.6 |
| `CH_SLOPE` | 38 | 17 | 0.4880 | 8 | 5.15 |
| `CH_PEAK_SHAPE` | 43 | 11 | 0.4519 | 7 | 3.39 |
| `TH_COUPLING` | 107 | 14 | 0.5903 | 11 | 2.99e+05 |
| `TCH_COUPLING` | 3 | 1 | 1.1354 | 8 | 10.1 |

保存的 `shape_checks` 允许按温度端点、盐内部峰、膜内部峰及交互符号做粗响应分组；这不是按完整拟合曲线或导数距离的聚类。出现最多的组如下：

| 粗响应签名 | 不同数值可接受方程 |
|---|---:|
| `T_all|Cpeak_all|Hpeak_none|CH_zero` | 23 |
| `T_all|Cpeak_all|Hpeak_all|CH_zero` | 18 |
| `T_all|Cpeak_all|Hpeak_all|CH_negative` | 15 |
| `T_none|Cpeak_all|Hpeak_none|CH_zero` | 8 |
| `T_all|Cpeak_all|Hpeak_partial|CH_negative` | 7 |
| `T_all|Cpeak_none|Hpeak_none|CH_zero` | 4 |
| `T_none|Cpeak_none|Hpeak_none|CH_zero` | 2 |
| `T_none|Cpeak_none|Hpeak_partial|CH_negative` | 2 |
| `T_all|Cpeak_none|Hpeak_partial|CH_negative` | 2 |
| `T_all|Cpeak_all|Hpeak_all|CH_positive` | 1 |

四轮都保留 Arrhenius 温度主族及盐峰竞争主族，属于同一数据上的逐轮适应性延续，不能记为四次独立重复发现。液膜响应由独立对称峰转向盐依赖联合峰，精确子型尚不稳定。

第六轮仅 33 个被筛选结构做了 23 折复核，22 个数值可接受；Pareto 前沿只对该复核子集成立。最终七参数式误差为 0.451943，全数据多初值求解 6/6 可行、雅可比条件数 3.39，六个非截距参数跨折符号一致；但膜峰位置只匹配 8/24，交互残差约 0.624 log 单位。求解稳定与形状充分是不同问题。因此可称温度激活与盐峰竞争是当前搜索内的稳健方向，不可称完整数学形式或交互机理收敛。

## 充分性边界与下一步

1. 当前库系统覆盖了温度、盐、液膜及交互的多种曲率、尾部和局部形状；但整体方程主要采用 log 电流加性分解。没有比较不同的电流/通量合成语法，函数空间充分性未建立。
2. 14 条后期模块缺少模块级来源等级与原式迁移记录。可补录为诊断启发或经验对照，不能倒填成文献原式。
3. 存档 `shape_checks` 只有拟合后峰与端点的粗描述；没有保存所有候选的完整曲线和导数。故本报告给出行为签名与族级汇总，未声称完成完整响应曲线聚类。
4. 若后续新增不同合成语法或其他家族，必须拟合并按相同预算复核；本审计没有重新拟合，也不能评价未生成的形式。
5. 同一批 168 条观测与 23 个开发折被反复使用。报告的族级精度和稳定性都是内部开发证据，不是外部预测或科学机理证明。
