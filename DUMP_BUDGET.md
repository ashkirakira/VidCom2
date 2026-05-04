# DUMP_BUDGET 使用说明

用于可视化 VidCom2 在 Video-MME 上每帧的 token 保留比例。通过 `DUMP_BUDGET` 环境变量在推理过程中写出 per-frame budget JSON + 帧图片，不改推理逻辑。

## 使用前提（严格要求）

使用 `DUMP_BUDGET=1` 时，**必须同时满足**以下所有条件：

1. **`--batch_size 1`**
   - dump 代码用 `doc_id[0]` / `video_inputs[0]` 取当前样本
   - `DUMP_BUDGET_FILTER` 也用 `doc_id[0]` 取 videoID 做白名单匹配
   - batch_size > 1 时只会 dump 每个 batch 的第一个样本，后面的样本被丢弃

2. **`COMPRESSOR=vidcom2`**
   - 帧图片由 `chat/qwen3_vl.py` 保存，曲线数据由 `token_compressor/vidcom2/models/qwen3_vl.py` 保存
   - 如果没开 vidcom2（baseline 或其他压缩方法），`Qwen3VLModel_forward` 不会被 patch，`budget.json` 不会被写

**违反以上任一条件，输出数据会是错的或缺失的。**

**关于多进程（`--num_processes=N`）：安全。**
每个进程是独立的 Python 解释器，各自维护自己的 `_current_video_id`，
accelerate 按 rank 把数据集切成不相交子集分给各进程，不会串位。
单卡装不下 8B 模型时请用 `--num_processes=8`（和 baseline 实验保持一致）。

## 正确用法

```bash
source /mnt/cpfs/gaoyizhuo-20260417/setup_env.sh
export COMPRESSOR=vidcom2 R_RATIO=0.25
export DUMP_BUDGET=1 DUMP_BUDGET_DIR=./budget_data/

accelerate launch --num_processes=8 -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2,max_num_frames=32 \
  --tasks videomme \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix dump_budget \
  --output_path ./logs/
```

## 只 dump 部分 videoID（DUMP_BUDGET_FILTER）

`DUMP_BUDGET_FILTER` 指向一个文本文件（每行一个 videoID）。命中白名单的样本走完整推理 + dump，其他样本直接跳过 `model.generate`，进度条前进但不做任何 GPU 工作。

适合"只关心 139 个退化视频"的场景，避免跑全量 900 道题。

```bash
export DUMP_BUDGET=1
export DUMP_BUDGET_FILTER=./degraded_video_ids.txt   # 每行一个 videoID
# 其他参数同上
```

注意：lmms-eval 最后会算评测指标，跳过的样本回答是空字符串，会被算作错误——所以**指标不可信**，只看 budget_data/ 里 dump 的内容就行。

## 禁止用法

- **baseline 实验（`COMPRESSOR` 不设）+ `DUMP_BUDGET=1`**：不会生成曲线数据，只有帧图片。无意义。
- **`--batch_size 2` 或更大 + `DUMP_BUDGET=1`**：只会 dump 每个 batch 的第一个样本。

## 输出

```
budget_data/
├── <videoID_1>/
│   ├── budget.json          # {scales, ks, frame_tokens, r_ratio}
│   ├── frame_000.jpg
│   ├── ...
│   └── frame_031.jpg
├── <videoID_2>/
│   └── ...
```

`budget.json` 字段：
- `scales`：每帧保留比例，长度 = 帧数
- `ks`：每帧保留 token 数 = round(scales × frame_tokens)
- `frame_tokens`：单帧经 spatial_merge 后的 token 总数
- `r_ratio`：基础保留比例（=0.25）

## 正常实验不受影响

不设 `DUMP_BUDGET`（或设为空字符串）时，所有 dump 代码都不会执行。正常 baseline / vidcom2 / 其他压缩方法的实验正常跑，零额外开销。

## 涉及的代码

- `lmms_eval/models/chat/qwen3_vl.py` — `generate_until` 中保存帧图片 + 设置 `_current_video_id`
- `token_compressor/vidcom2/models/qwen3_vl.py` — `Qwen3VLModel_forward` 压缩循环中收集 scales + 写 `budget.json`
- `plot_budget_overlay_template.py` — 读 `budget_data/` 生成可视化
