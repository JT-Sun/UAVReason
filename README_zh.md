# UAVReason

**Can Vision-Language Models Think from the Sky? Unifying UAV Reasoning and Generation**

UAVReason 是一个面向无人机俯视视角的多模态推理与生成数据集，基于 UAVScenes RGB 图像构建，并提供 VQA / Caption 标注和深度数据，可用于 UAV 场景理解、空间推理、时序推理、航向推理、深度感知和跨模态生成等任务。

> 请根据本地环境替换 README 中的所有路径。

## Links

- Paper: https://arxiv.org/abs/2604.05377
- Code: https://github.com/JT-Sun/UAVReason
- UAVReason HF repo: https://huggingface.co/datasets/jarvissun/UAVReason
- UAVReason depth: https://huggingface.co/datasets/jarvissun/UAVReason_depth
- UAVScenes: https://github.com/sijieaaa/UAVScenes

> 如果 Hugging Face 仓库后续重命名或拆分，请将上述链接替换为最新地址。

## Data Components

| Data | Format | Usage |
|---|---|---|
| Single-frame VQA | LLaVA-style JSONL | 单帧问答、计数、属性、空间关系推理 |
| Two-frame VQA | LLaVA-style JSONL | 双帧时序变化、距离变化、关系变化推理 |
| Heading VQA | LLaVA-style JSONL | UAV 航向 / 运动方向推理 |
| Scene Caption | LLaVA-style JSONL | UAV 场景描述 |
| Depth | `.npy` / `_depth_vis.png` / stats | 深度监督、深度可视化、RGB-Depth 生成 |
| Generation data | i2i JSONL → parquet | BAGEL `unified_edit` 跨模态生成训练 |

## Directory Structure

```bash
UAVReason/
├── UAVScenes/
│   └── interval5_CAM_LIDAR/
│       ├── interval5_AMtown01/
│       ├── interval5_AMtown02/
│       └── ...
├── UAVReason_depth/
│   ├── interval5_AMtown01/
│   │   ├── 1658137057.641204937_depth.npy
│   │   ├── 1658137057.641204937_depth_vis.png
│   │   └── 1658137057.641204937_stats.json
│   └── ...
├── annotations/
│   ├── llava_vqa_single_1f_anchor_train.jsonl
│   ├── llava_vqa_temporal_2f_anchor_train.jsonl
│   ├── llava_vqa_temporal_2f_IHeading_train.jsonl
│   ├── llava_vqa_scene_caption.jsonl
│   └── ...
├── i2i_jsonl/
│   ├── uav_rgb2depth.jsonl
│   ├── uav_rgb2seg.jsonl
│   ├── uav_d_text2rgb.jsonl
│   ├── uav_seg_text2rgb.jsonl
│   ├── uav_dseg_text2rgb.jsonl
│   └── uav_recon.jsonl
└── parquet/
    └── uav_unified_edit/
```

## VQA / Caption Format

VQA 与 Caption 使用 LLaVA-style JSONL。每一行是一个样本：

```json
{
  "image": [
    "UAVScenes/interval5_CAM_LIDAR/interval5_AMtown02/interval5_CAM/1658133165.089699441.jpg"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nIn this UAV frame, north is approximately towards the top-right of the image. Answer concisely. How many roofs are visible in the scene?"
    },
    {
      "from": "gpt",
      "value": "There are 5 roofs visible in the scene."
    }
  ],
  "meta": {
    "task": "uav_vqa_1f",
    "scene": "interval5_AMtown02",
    "stem": "1658133165.089699441",
    "category": "Common Scenes",
    "subtype": "B-Count"
  }
}
```

说明：

- Single-frame VQA：`image` 中包含 1 张图像。
- Two-frame VQA / Heading VQA：`image` 中包含 2 张图像，顺序为 `Image-1 → Image-2`。
- Scene Caption：输入 UAV 图像，输出场景描述。
- `meta` 用于任务划分、类别统计和评估分析。

## Depth Data

深度数据包含：

```bash
{stem}_depth.npy
{stem}_depth_vis.png
{stem}_stats.json
```

`.npy` 为原始深度数组，`_depth_vis.png` 为灰度深度可视化图，`_stats.json` 为深度统计信息。BAGEL 训练中，depth 作为图像模态使用：默认优先使用 `.npy`，并在数据读取阶段转换为灰度 depth image；若 `.npy` 不存在，可回退使用 `_depth_vis.png`。

## Using with BAGEL

| UAVReason data | BAGEL branch | Format |
|---|---|---|
| VQA / Caption | `vlm_sft` | LLaVA-style JSONL |
| RGB / Depth / Segmentation generation | `unified_edit` | parquet |

### Register VQA / Caption

在 BAGEL `data/dataset_info.py` 中注册：

```python
"vlm_sft": {
    "uav_vqa_1f": {
        "data_dir": "/path/to/UAVReason",
        "jsonl_path": "/path/to/annotations/llava_vqa_single_1f_anchor_train.jsonl",
        "num_total_samples": 172037,
    },
    "uav_vqa_2f": {
        "data_dir": "/path/to/UAVReason",
        "jsonl_path": "/path/to/annotations/llava_vqa_temporal_2f_anchor_train.jsonl",
        "num_total_samples": 57462,
    },
    "uav_vqa_iheading": {
        "data_dir": "/path/to/UAVReason",
        "jsonl_path": "/path/to/annotations/llava_vqa_temporal_2f_IHeading_train.jsonl",
        "num_total_samples": 57456,
    },
    "uav_vqa_scene_caption": {
        "data_dir": "/path/to/UAVReason",
        "jsonl_path": "/path/to/annotations/llava_vqa_scene_caption.jsonl",
        "num_total_samples": 19903,
    },
}
```

如果 JSONL 中图像路径是绝对路径，可将 `data_dir` 设置为 `/`；如果是相对路径，`data_dir` 应指向包含 `UAVScenes/` 的目录。

### Build Generation Parquet

生成任务先整理为 ShareGPT-style image-to-image JSONL，再转换为 BAGEL `unified_edit` parquet：

```bash
python uav_jsonl_to_parquet_for_bagel.py \
  --in_jsonl \
    i2i_jsonl/uav_rgb2depth.jsonl \
    i2i_jsonl/uav_rgb2seg.jsonl \
    i2i_jsonl/uav_d_text2rgb.jsonl \
    i2i_jsonl/uav_seg_text2rgb.jsonl \
    i2i_jsonl/uav_dseg_text2rgb.jsonl \
    i2i_jsonl/uav_recon.jsonl \
  --out_dir parquet/uav_unified_edit \
  --shard_size 5000 \
  --prefer_depth npy \
  --depth_npy_root /path/to/UAVReason_depth \
  --depth_vis_root /path/to/UAVReason_depth \
  --depth_npy_template "{scene}/{stem}_depth.npy" \
  --overwrite
```

输出 parquet 包含：

```text
text        instruction
image_list  source image path(s) + target image path
task        task name
```

`image_list` 中最后一张图像作为生成监督目标。

### Register Generation Data

```python
"unified_edit": {
    "uav_unified_edit": {
        "data_dir": "/path/to/parquet/uav_unified_edit",
        "num_files": 32,
        "num_total_samples": 159224,
        "parquet_info_path": "/path/to/parquet/uav_unified_edit/parquet_info.json",
    }
}
```

如果重新生成 parquet，请根据实际 `parquet_info.json` 更新 `num_files` 和 `num_total_samples`。

## Supported Tasks

UAVReason 可用于但不限于：

- UAV visual question answering
- UAV scene captioning
- Single-frame spatial reasoning
- Two-frame temporal reasoning
- UAV heading / motion direction reasoning
- RGB → Depth generation
- RGB → Segmentation generation
- Depth / Segmentation / Text → RGB generation
- RGB / Depth / Segmentation reconstruction
- UAV multimodal model adaptation and evaluation

## BAGEL Joint SFT Example

```bash
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/uav_mix.yaml \
  --model_path /path/to/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from /path/to/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --visual_und True \
  --visual_gen True \
  --results_dir results/uavreason_bagel \
  --checkpoint_dir results/uavreason_bagel/checkpoints \
  --lr 2e-5 \
  --num_workers 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240
```

## Citation

```bibtex
@article{sun2026uavreason,
  title={UAVReason: A Unified, Large-Scale Benchmark for Multimodal Aerial Scene Reasoning and Generation},
  author={Sun, Jintao and Zhang, Hu and Di, Donglin and Ding, Gangyi and Zheng, Zhedong},
  journal={arXiv preprint arXiv:2604.05377},
  year={2026}
}
```
