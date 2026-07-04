# UAVReason

**UAVReason: A Unified, Large-Scale Benchmark for Multimodal Aerial Scene Reasoning and Generation**

UAVReason is a multimodal aerial scene reasoning and generation dataset for UAV-view images. It is built on UAVScenes RGB images and provides VQA / caption annotations and additional depth data. The dataset can be used for UAV visual question answering, scene captioning, spatial reasoning, temporal reasoning, heading reasoning, depth-aware perception, and cross-modal generation.

This repository provides the data usage guide and BAGEL data adaptation scripts.

## Links

- Paper: https://arxiv.org/abs/2604.05377
- Code: https://github.com/JT-Sun/UAVReason
- UAVReason annotations: https://huggingface.co/datasets/jarvissun/UAVReason
- UAVReason depth: https://huggingface.co/datasets/jarvissun/UAVReason_depth
- UAVScenes: https://github.com/sijieaaa/UAVScenes

Please refer to the Hugging Face pages for the latest released files. If the dataset repositories are renamed or reorganized, replace the links above with the latest URLs.

## Documentation

For detailed data download, directory structure, and BAGEL configuration, please see:

- [Chinese Usage Guide](README_zh.md)
- [English Usage Guide](README_en.md)

## Data Components

UAVReason currently contains four main JSONL annotation types:

| Data | Format | Description |
|---|---|---|
| Single-frame VQA | LLaVA-style JSONL | Single-image UAV question answering and spatial reasoning |
| Two-frame VQA | LLaVA-style JSONL | Temporal change and relation reasoning between two UAV frames |
| Heading VQA | LLaVA-style JSONL | UAV heading / motion direction reasoning |
| Scene Caption | LLaVA-style JSONL | UAV scene caption generation |

Additional depth data is provided separately:

| Data | Format | Description |
|---|---|---|
| Depth array | `.npy` | Original depth array |
| Depth visualization | `_depth_vis.png` | Grayscale depth visualization |
| Depth statistics | `_stats.json` | Depth metadata and statistics |

## Using UAVReason with BAGEL

UAVReason is adapted to BAGEL through two branches:

| UAVReason data | BAGEL branch | Format |
|---|---|---|
| VQA / Caption | `vlm_sft` | LLaVA-style JSONL |
| RGB / Depth / Segmentation generation | `unified_edit` | parquet |

The VQA and caption files can be directly registered in BAGEL `vlm_sft`.

For generation tasks, we first organize RGB, depth, segmentation, and text-conditioned samples as ShareGPT-style image-to-image JSONL files. Then we convert them into BAGEL `unified_edit` parquet using:

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

Please replace all paths with your local paths.

The script `uav_jsonl_to_parquet_for_bagel.py` converts image-to-image JSONL files into parquet shards with the following fields:

```text
text        instruction
image_list  source image path(s) + target image path
task        task name
```

The last image in `image_list` is used as the target image. For depth-related tasks, the pipeline prefers `.npy` depth files by default and can fall back to `_depth_vis.png` when needed.

## Supported Tasks

UAVReason can be used for:

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

## Suggested Files in This Repository

A minimal release should include:

```text
README.md
README_zh.md
README_en.md
uav_jsonl_to_parquet_for_bagel.py
uav_mix.yaml
dataset_info_uav_example.py
```

Optionally, add a small `examples/` folder with several VQA JSONL and image-to-image JSONL samples so that users can quickly check the expected format.

## Citation

```bibtex
@article{sun2026uavreason,
  title={UAVReason: A Unified, Large-Scale Benchmark for Multimodal Aerial Scene Reasoning and Generation},
  author={Sun, Jintao and Zhang, Hu and Di, Donglin and Ding, Gangyi and Zheng, Zhedong},
  journal={arXiv preprint arXiv:2604.05377},
  year={2026}
}
```
