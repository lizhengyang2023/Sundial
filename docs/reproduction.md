# Sundial 复现流程：三源预训练与 ETT/Weather 零样本评估

## 文档目录

- [Sundial 论文](Sundial.pdf): 模型、TimeFlow 和实验设置的主要依据；
- [Timer 论文](Timer.pdf): 补足 S3 数据处理方式(将多变量序列拆分为单变量)。数据源包括 UTSD, Chronos, LOTSA。

## 一条数据怎样走到预测

1. 源数据进入 S3。按文件惰性取得本地或 Hugging Face 上的 Arrow/Parquet，用 `datasets` 流式读取记录；数值向量直接提取，二维矩阵按变量轴拆成独立一维序列，跳过时间戳等元数据。模型能逐变量预测，却不会显式学习变量之间的关联。每条变量序列按时间前 90% / 后 10% 分成预训练 / 验证段，用训练段的统计量清洗和标准化，避免验证段信息反向进入训练段。Timer §3.2 明确给出逐变量 9:1 划分及训练段统计量归一化；Sundial §4 只明确逐变量归一化和变长窗口。

2. 存储与抽样。清洗后的每条变量存为一条 Parquet 记录，按源分片，分片内随机打乱。训练时对 UTSD、Chronos、LOTSA 各取 1/3 的样本；在源内等概率选择一条变量序列，再随机选起点和历史长度。一个 batch 的大小必须是 3 的倍数，确保每批三源条数相同。这样控制的是来源权重，同一来源下的变量条目等概率；长度大的变量可以在不同随机窗口反复使用。论文仅称“按预设比例跨领域采样”和“全局打乱 Parquet”，没有公开具体比例，因此 1:1:1 是本项目选择。

3. 输入和目标。历史最长 2880 点，随机取变长历史。模型每 16 点生成一个 patch；不足 16 的开头从左侧补零，并携带有效位掩码。一个训练样本可让多个 patch 提供监督：第 `i` 个位置的条件 $h_i$ 预测该 patch 之后最多 `F=720` 点。足够长的原序列只抽取有完整 720 点未来的窗口；较短的原序列允许末尾目标不足 720 点，但用掩码排除补零。

4. 历史重归一化。每次前向只用当前有效历史点求均值和标准差，把历史及目标变换到该局部尺度；采样结束再还原。它和第 1 步的全序列逐变量归一化是不同层次。模型生成时允许连续值，不做离散分桶。

5. Transformer 得到条件。每个 patch 的数值和掩码拼接，经 MLP 投影。12 层 decoder-only Transformer 使用因果注意力、RoPE 和 Pre-LN。$h_i$ 汇总到第 `i` 个 patch 为止的历史，后面的真实值无法作为注意力输入；它是未来分布的条件，不是预测序列本身。

6. TimeFlow 学未来分布。真实未来 `y` 和标准高斯噪声 `ε` 的线性路径是 $y_t=(1-t)ε+ty$，`t∼U[0,1]`。给 FM-Net 输入 $(y_t,t,h_i)$，让其输出速度 $v_θ$，最小化 $||v_θ-(y-ε)||^2$（只在有效目标点上）。训练时每个目标只抽一次 `t` 和 `ε`，只运行一次 FM-Net；梯度同时更新 Transformer 和 FM-Net。

7. 推理与测评。固定历史的 `h`，从新高斯噪声出发，按 $y←y+v_θ(y,t,h)/50$ 运行 50 次，得到一条可能的未来。多次换噪声得到多个样本，取均值或分位数；预测长度小于 720 就截断。ETTh1/2、ETTm1/2、Weather 只在各自测试段评估，不进入任何预训练文件。

## 结构、参数与技术栈

默认实现 [model.py](../src/sundial/model.py) 对应 Sundial 论文附录 B 表 5 的 Base、F=720：patch 16、最大上下文 2880、Transformer 12 层、`D=768`、FFN 3072、12 头、TimeFlow 宽度 768 且 3 个 AdaLN 残差层、采样 50 步。当前参数量 128,329,680，与论文的 128M 标称相符。另一个论文实验用 `F=16`，面向 FEV；这里仅做 ETT/Weather 长期预测，因此用 720。

技术栈是 Python、PyTorch（注意力用 `scaled_dot_product_attention`，训练数据用 `IterableDataset` / `DataLoader`）、Hugging Face `datasets`、NumPy、pandas、PyArrow/Parquet。训练采用 AdamW，CUDA 上自动尝试 bfloat16，否则 float16；支持梯度累积和 checkpoint。学习率、训练总步数、warmup、weight decay、异常值阈值并非论文公开的确定数值，CLI 中只是可调整的起点。论文使用 32 张 A100；当前脚本是单进程训练入口，算力、吞吐和最终权重不能直接与论文等同。

还有一个复现边界：论文式 (6)–(9) 写的是预测速度 `y-ε`；[官方发布的 `flow_loss.py`](https://huggingface.co/thuml/sundial-base-128m/blob/main/flow_loss.py) 实际让网络预测干净目标 `y`，推理时用 `pred-初始噪声` 作为更新方向，并加入位置权重等细节。本项目选择论文公式，因此架构与参数量对齐，但 checkpoint 不兼容官方权重，也不保证逐数值复现其表格。

## 下载和转换

数据来源：[UTSD](https://huggingface.co/datasets/thuml/UTSD)、[Chronos datasets](https://huggingface.co/datasets/autogluon/chronos_datasets)、[LOTSA](https://huggingface.co/datasets/Salesforce/lotsa_data)。现在可用 `hf://datasets/组织/仓库/子目录` 输入：远端目录按页枚举，仅取 `.arrow` / `.parquet`，处理到某个文件时才下载它到 `--hf-cache`。本地目录也用迭代式文件发现，忽略 JSON。UTSD 先指定 `UTSD-1G`，避免把 1G/2G/4G/12G 层级重复计数。Chronos 的 `train` 分片可能包含整条时序，不能凭分片名认定已剔除 ETT 测试段；仍需审查来源目录与条目 ID。

```powershell
python -m pip install -e .
python -m sundial.cli prepare --source utsd --input hf://datasets/thuml/UTSD/UTSD-1G --output corpus --hf-cache .cache/huggingface --read-batch-size 64

# 其余来源在选择具体子目录后使用同一入口；不要直接对 800 GB 以上仓库执行全量 prepare。
# python -m sundial.cli prepare --source chronos --input hf://datasets/autogluon/chronos_datasets/<子目录> --output corpus
# python -m sundial.cli prepare --source lotsa --input hf://datasets/Salesforce/lotsa_data/<子目录> --output corpus
```

`prepare` 只发现 `.arrow`、`.parquet`，按文件下载并流式读取记录，转换后的序列攒满 `--shard-rows` 条就写一个 Parquet 分片。处理整个远端目录仍会依次下载其中全部匹配文件，因此先用 UTSD-1G 试验。训练时按来源等比例抽序列和随机滚动窗口，分片读取采用 LRU 缓存。异常值处理采用“训练段中位数 ±10 MAD 尺度”截断，缺失值以前向填充、开头以训练段中位数填充；这是工程选择，论文没有披露相同阈值或插补算法。`corpus/<source>/manifest.json` 记录转换条数与跳过数量，训练前应核对。

## 训练与评估命令

```powershell
# 先只用 UTSD-1G 验证数据与训练链路；批大小可为任意正整数。
python -m sundial.cli train --corpus corpus --sources utsd --steps 100 --batch-size 6 --workers 2 --checkpoint checkpoints/utsd-pilot.pt

# 三源准备完成后，默认按 1:1:1 抽样；多 worker 并行生成 batch。
python -m sundial.cli train --corpus corpus --steps 100000 --batch-size 6 --accum 4 --checkpoint checkpoints/sundial-base.pt

# 中断后按总步数继续；--steps 表示期望到达的总步数。
python -m sundial.cli train --corpus corpus --steps 100000 --batch-size 6 --accum 4 --resume checkpoints/sundial-base.pt --checkpoint checkpoints/sundial-base.pt

# 论文长期预测任务：horizon 96/192/336/720，最长历史 2880。
python -m sundial.cli evaluate --checkpoint checkpoints/sundial-base.pt --data dataset --samples 20 --flow-steps 50 --stride 1 --output results/zero_shot.json
```

评估使用 Time-Series-Library 常见的 ETT 固定时间边界与 Weather 70%/10%/20% 边界；每个变量用该测评数据的训练段统计量标准化，报告标准化尺度的 MSE/MAE。评估窗口不参与模型更新；`--stride 1` 遍历所有可用测试窗口，代价较高，初次检查可把 stride 调大。采样 `--samples 20` 时，点预测采用生成样本的均值；论文没有清楚规定表 1 的所有点预测聚合细节，因此应在比较成绩时固定并注明这个选择。当前脚本只实现请求的五个数据集和 MSE/MAE，没有 GIFT-Eval/FEV 的概率指标。

建议的验收次序：检查三个 manifest 的数据条数和泄漏过滤情况 → 小配置跑通一个 batch 并确认损失有限、参数更新 → 用少量测试窗口核对输出形状与逆归一化 → 开始完整预训练 → 用上述固定设置输出 5×4 组 MSE/MAE。若更换语料子集、训练步数或抽样方式，应保存命令、随机种子与数据版本，避免和论文的原始 1032B 结果混称。

## 相关资源

资源	            链接
论文	            https://arxiv.org/abs/2502.00816
GitHub	            https://github.com/thuml/Sundial
HuggingFace 模型	https://huggingface.co/thuml/sundial-base-128m
GGUF 量化版	        https://huggingface.co/amaye15/sundial-gguf
GIFT-Eval 基准	    https://huggingface.co/spaces/Salesforce/GIFT-Eval
Time-Series-Library	https://github.com/thuml/Time-Series-Library
中文解读	         https://mp.weixin.qq.com/s/y3sc2e2lmW1sqfnoK-ZdDA
