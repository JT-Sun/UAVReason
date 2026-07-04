# UAVReason

**UAVReason: A Unified, Large-Scale Benchmark for Multimodal Aerial Scene Reasoning and Generation**

UAVReason is a multimodal aerial scene reasoning and generation dataset for UAV-view images. It is built on UAVScenes RGB images and provides VQA / caption annotations, image-to-image generation JSONL files, and additional depth data. The dataset can be used for UAV visual question answering, scene captioning, spatial reasoning, temporal reasoning, heading reasoning, depth-aware perception, and cross-modal generation.

This repository provides the data usage guide and BAGEL data adaptation scripts. 

## Links

- Paper: https://arxiv.org/abs/2604.05377
- Code: https://github.com/JT-Sun/UAVReason
- UAVReason VQA / Caption / Generation JSONL: https://huggingface.co/datasets/jarvissun/UAVReason_vqa/tree/main
- UAVReason depth: https://huggingface.co/datasets/jarvissun/UAVReason_depth
- UAVScenes: https://github.com/sijieaaa/UAVScenes

Please refer to the Hugging Face pages for the latest released files. If the dataset repositories are renamed or reorganized, replace the links above with the latest URLs.

## Documentation

For detailed data download, directory structure, and BAGEL configuration, please see:

- [Chinese Usage Guide](README_zh.md)
- [English Usage Guide](README_en.md)

## Data Components

UAVReason contains four main VQA / caption JSONL annotation types:

| Data | Format | Description |
|---|---|---|
| Single-frame VQA | LLaVA-style JSONL | Single-image UAV question answering and spatial reasoning |
| Two-frame VQA | LLaVA-style JSONL | Temporal change and relation reasoning between two UAV frames |
| Heading VQA | LLaVA-style JSONL | UAV heading / motion direction reasoning |
| Scene Caption | LLaVA-style JSONL | UAV scene caption generation |

UAVReason also provides image-to-image JSONL files for generation tasks. These files are used to build BAGEL `unified_edit` parquet shards:

| Data | Format | Description |
|---|---|---|
| RGB -> Depth | ShareGPT-style i2i JSONL | Generate a depth map from an RGB UAV image |
| RGB -> Segmentation | ShareGPT-style i2i JSONL | Generate a semantic segmentation map from an RGB UAV image |
| Depth + Text -> RGB | ShareGPT-style i2i JSONL | Generate an RGB image conditioned on depth and text |
| Segmentation + Text -> RGB | ShareGPT-style i2i JSONL | Generate an RGB image conditioned on segmentation and text |
| Depth + Segmentation + Text -> RGB | ShareGPT-style i2i JSONL | Generate an RGB image conditioned on depth, segmentation, and text |
| Reconstruction | ShareGPT-style i2i JSONL | RGB / depth / segmentation reconstruction |

Additional depth data is provided separately:

| Data | Format | Description |
|---|---|---|
| Depth array | `.npy` | Original depth array |
| Depth visualization | `_depth_vis.png` | Grayscale depth visualization |
| Depth statistics | `_stats.json` | Depth metadata and statistics |

## Recommended Directory Structure

```bash
UAVReason/
├── UAVScenes/                         # RGB images from UAVScenes
│   └── interval5_CAM_LIDAR/
│       ├── interval5_AMtown01/
│       ├── interval5_AMtown02/
│       └── ...
│
├── UAVReason_depth/                   # Depth files
│   ├── interval5_AMtown01/
│   │   ├── 1658137057.641204937_depth.npy
│   │   ├── 1658137057.641204937_depth_vis.png
│   │   └── 1658137057.641204937_stats.json
│   └── ...
│
├── annotations/                       # VQA / Caption JSONL
│   ├── llava_vqa_single_1f_anchor_train.jsonl
│   ├── llava_vqa_temporal_2f_anchor_train.jsonl
│   ├── llava_vqa_temporal_2f_IHeading_train.jsonl
│   ├── llava_vqa_scene_caption.jsonl
│   └── ...
│
├── i2i_jsonl/                         # Generation JSONL
│   ├── uav_rgb2depth.jsonl
│   ├── uav_rgb2seg.jsonl
│   ├── uav_d_text2rgb.jsonl
│   ├── uav_seg_text2rgb.jsonl
│   ├── uav_dseg_text2rgb.jsonl
│   └── uav_recon.jsonl
│
└── parquet/
    └── uav_unified_edit/
```

Please replace all paths according to your local environment.

## VQA / Caption Format

VQA and caption annotations follow the LLaVA-style JSONL format. Each line is one sample:

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

Notes:

- Single-frame VQA contains one image.
- Two-frame VQA and Heading VQA contain two images in the order `Image-1 -> Image-2`.
- Caption data uses UAV images as input and scene descriptions as output.
- `meta` is used for task grouping, category-level statistics, and evaluation.

## Depth Data

Depth data contains:

```bash
{stem}_depth.npy
{stem}_depth_vis.png
{stem}_stats.json
```

- `.npy`: original depth array.
- `_depth_vis.png`: grayscale depth visualization rendered from the depth array.
- `_stats.json`: depth statistics.

For BAGEL training, depth is used as an image modality. By default, the pipeline prefers `.npy` depth files and converts them into grayscale depth images during data loading. If `.npy` is unavailable, `_depth_vis.png` can be used as fallback.

## Using UAVReason with BAGEL

We use two BAGEL branches:

| UAVReason data | BAGEL branch | Format |
|---|---|---|
| VQA / Caption | `vlm_sft` | LLaVA-style JSONL |
| RGB / Depth / Segmentation generation | `unified_edit` | parquet |

### 1. Register VQA / Caption

Add the following entries to `data/dataset_info.py`:

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

If image paths in JSONL are absolute paths, `data_dir` can be set to `/`. If image paths are relative paths, `data_dir` should point to the directory that contains `UAVScenes/`.

### 2. Build Generation Parquet

Generation samples are first organized as ShareGPT-style image-to-image JSONL files and then converted into BAGEL `unified_edit` parquet:

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

The output parquet contains:

```text
text        instruction
image_list  source image path(s) + target image path
task        task name
```

The last image in `image_list` is used as the target image.

### 3. Register Generation Data

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

If you regenerate the parquet shards, update `num_files` and `num_total_samples` according to the actual `parquet_info.json`.

## Supported Tasks

UAVReason can be used for, but is not limited to:

- UAV visual question answering
- UAV scene captioning
- Single-frame spatial reasoning
- Two-frame temporal reasoning
- UAV heading / motion direction reasoning
- RGB -> Depth generation
- RGB -> Segmentation generation
- Depth / Segmentation / Text -> RGB generation
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
