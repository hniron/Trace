# Trace 统计表格使用说明

本仓库是 **Qwen3-Next-80B-A3B-Instruct** 大模型推理时 HBM 访存 trace 的统计分析结果。

## 一、数据来源与背景

### 1.1 采集目标

为 FPGA 硬件仿真平台提供真实大模型推理的 HBM 访问 trace 作为输入源。FPGA 侧分两块：Host FPGA 模拟 CPU/GPU 发起访存请求，HBM FPGA 模拟 HBM 的行为与时序。

### 1.2 采集方法

| 项目 | 说明 |
|---|---|
| 模型 | Qwen3-Next-80B-A3B-Instruct（NVFP4 量化，MoE：512 专家/激活 10，48 层混合线性注意力+全注意力） |
| GPU | NVIDIA RTX 6000D（Blackwell，sm_120，96GB） |
| 采集工具 | NVBit 1.8.1（动态二进制插桩，SASS 指令级） |
| 推理框架 | vLLM 0.33（`enforce_eager=True` 关闭 CUDA Graph） |
| 采集范围 | decode 阶段 3 个 token（预热后，只抓 decode） |
| 数据规模 | 2.29 亿条访存记录，原始 trace 7.33 GB |

### 1.3 原始 trace 记录格式

每条访存记录 28 字节（对齐到 32 字节），字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| grid_launch_id | u64 | kernel 启动序号（每次 kernel launch 递增） |
| cta_id_x | u32 | 发起访问的 block（CTA）在 x 维度的编号 |
| warp_id | u16 | 发起访问的 warp 在 block 内的编号 |
| opcode_id | u16 | 访存指令类型编号（映射见 `opcode_stats.csv`） |
| addr | u64 | **访问的地址**（warp 首个活跃线程的 GPU 虚拟地址） |
| size | u32 | 每线程的访存字节数（1/2/4/8/16 字节） |

> **关于地址**：`addr` 是 GPU 的统一虚拟地址（UVA）。它不是物理地址，但完整记录了访问的地址空间布局与访问模式，足以支撑 FPGA 侧做地址相关的仿真。

## 二、文件清单

| 文件 | 类型 | 内容 |
|---|---|---|
| `README.md` | 文档 | 本说明 |
| `trace_analysis.xlsx` | Excel | 所有统计表格的多 sheet 合并版（含 6 个 sheet） |
| `summary.txt` | 文本 | 汇总报告 |
| `kernel_class_summary.csv` | CSV | kernel 按功能分类汇总 |
| `opcode_stats.csv` | CSV | 访存指令类型分布 |
| `size_stats.csv` | CSV | 访存大小分布 |
| `kernel_stats.csv` | CSV | 每次 kernel 启动的逐条统计 |
| `address_region_distribution.csv` | CSV | **地址区域分布**（512MB 桶，读写分开） |
| `hot_pages.csv` | CSV | **4KB page 级热点**统计 |

## 三、每个表格的详细说明

### 3.1 kernel_class_summary.csv —— kernel 分类汇总

把 9327 次 kernel 启动按**功能**归为 11 类后的汇总，回答"各类计算占多少访存"。

| 列名 | 含义 |
|---|---|
| class | kernel 功能分类（见下表） |
| records | 该类 kernel 产生的访存记录总数 |
| bytes_warp | warp 级访存总字节数（= size × 32 线程的累加，单位字节） |
| percentage | 该类记录数占全部记录的百分比 |

分类含义：

| 分类 | 对应计算 | 访存特征 |
|---|---|---|
| GEMM | 矩阵乘（CUTLASS，含 MoE 专家权重乘法） | 大块连续读权重 |
| GEMV | 矩阵-向量乘（decode 时 batch=1 的投影） | 连续读权重 |
| MoE | 专家路由（topkGating、expert prefix sum） | 稀疏、不规则访问 |
| LinearAttention(delta_rule) | Qwen3-Next 的线性注意力（GDN 层） | 固定大小递推状态流式读写 |
| Attention | 全注意力（Q/K/V、KV cache） | 分页 KV cache 随机访问 |
| Norm / Activation | 层归一化、激活函数 | 小粒度激活读写 |
| Elementwise/Copy | PyTorch 张量逐元素操作、拷贝 | 大量流式读写中间激活 |
| FP4Convert | FP16→FP4 量化转换 | 量化相关访存 |
| KVCache | KV cache 管理 | 分页管理 |

### 3.2 opcode_stats.csv —— 访存指令类型分布

统计每条访存记录对应的 SASS 指令类型。

| 列名 | 含义 |
|---|---|
| opcode_id | 指令类型编号（原始 trace 里 `opcode_id` 字段的值） |
| opcode_name | SASS 指令名 |
| records | 该指令出现的次数 |
| percentage | 占全部记录的百分比 |

**指令名读写规律**：

| 前缀 | 含义 |
|---|---|
| `LDG` | 从全局内存 Load（读） |
| `STG` | 向全局内存 Store（写） |
| `LD` / `LDL` | 从局部内存 Load（读，寄存器溢出/局部数组） |
| `STL` | 向局部内存 Store（写） |
| `ATOMG` / `REDG` | 全局内存原子/归约操作（读改写） |

后缀含义：`.E` = 普通全局访问；`.128`/`.64`/`.U16`/`.U8` 等 = 访问位宽；`.CONSTANT` = 常量内存；`.LTC128B` = 128 字节 L2 请求（TMA 相关）；`.STRONG.GPU`/`.STRONG.SM` = 不同一致性域；`.EF` = 特定内存语义。

### 3.3 size_stats.csv —— 访存大小分布

统计每次访存（每线程）的字节数。

| 列名 | 含义 |
|---|---|
| bytes_per_thread | 每线程单次访存的字节数 |
| records | 该大小的访存次数 |
| percentage | 占比 |

> 本数据中 4 字节（53%）和 2 字节（30%）占主导，这是 NVFP4 量化权重反量化 + MoE 专家 gather 的典型特征；8/16 字节是 GEMM 的向量化访存。

### 3.4 kernel_stats.csv —— 每次 kernel 启动的逐条统计

9327 次 kernel 启动，每次一行。

| 列名 | 含义 |
|---|---|
| grid_id | kernel 启动序号（对应原始 trace 的 `grid_launch_id`） |
| class | 该 kernel 的功能分类（同 3.1） |
| kernel_name | kernel 的 C++ 名字（可能很长） |
| records | 该次 kernel 启动产生的访存记录数 |
| bytes_warp | warp 级访存字节数 |

### 3.5 address_region_distribution.csv —— 地址区域分布

把全部访问地址按 **512MB 区间**分段统计，回答"访存集中在哪些地址段、各段读写比例"。

| 列名 | 含义 |
|---|---|
| region_start_hex | 地址段起始地址（16 进制） |
| region_start | 地址段起始地址（十进制） |
| read_count | 该地址段的读（Load）次数 |
| write_count | 该地址段的写（Store）次数 |
| atomic_count | 该地址段的原子操作次数 |
| other_count | 其他类型次数 |
| total_count | 总访问次数 |

**本数据的地址分布特征**：

| 地址段 | 占比 | 特征 |
|---|---|---|
| `0x7d5b00000000` | 48.3% | **极热区**：仅 16KB 范围被反复读 1.1 亿次（全读），疑似 MoE 路由表/共享索引 |
| `0x7d4c60000000` | 27.6% | 主权重区：读为主，含少量写 |
| `0x7d4e40000000` | 11.0% | 权重区（全读） |
| `0x7d4d20000000` | 1.4% | 激活写出区（几乎全写） |
| `0x0` 附近 | 0.3% | 低地址（pinned/unified memory） |

### 3.6 hot_pages.csv —— 4KB page 级热点

把地址按 **4KB 页**对齐后统计，列出访问最频繁的前 200 个 page，回答"哪些页最热、读写比例"。

| 列名 | 含义 |
|---|---|
| page_addr_hex | 页起始地址（16 进制，4KB 对齐） |
| page_addr | 页起始地址（十进制） |
| access_count | 该页被访问的总次数 |
| read_count | 读次数 |
| write_count | 写次数 |
| atomic_count | 原子操作次数 |

> 本数据中 `0x7d5b00000000`–`0x7d5b00003000` 这 4 个连续页（共 16KB）被读了约 1.1 亿次，是最极端的访问热点，可重点关注其行为。

## 四、关于"地址信息"的说明

地址信息**完整存在于原始 trace**（`addr` 字段，见 1.3），本仓库通过两个表格把它呈现出来：

- **宏观**：`address_region_distribution.csv`（512MB 段，看整体地址空间布局与读写）
- **微观**：`hot_pages.csv`（4KB 页，看热点页）

如果 FPGA 项目组需要**逐条地址回放**（而非统计），需要用到原始 trace 文件（`.bin`，7.33GB，本仓库未包含），格式见 1.3。原始 trace 可按需进一步聚合为 page 粒度（4KB/2MB）或按时间窗分组。

## 五、关键结论速览

1. **计算类型**：GEMM(64.4%) + GEMV(28.6%) 占绝对主导，是 HBM 压力的主要来源
2. **访存粒度**：4B(53%)、2B(30%) 为主 —— NVFP4 量化 + MoE 的典型特征
3. **读写比**：以读为主（权重读取），写集中在少量激活写出区域
4. **地址热点**：存在一个 16KB 的极热读区（`0x7d5b00000000`），占 48% 访问，值得 FPGA 侧重点仿真
5. **覆盖完整性**：注意力权重、MoE、KV cache 三类 FPGA 项目组要求的访存模式均已覆盖
