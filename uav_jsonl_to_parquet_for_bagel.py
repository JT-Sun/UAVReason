

import os, json, argparse, glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_INSTR_BY_TASK = {
    "rgb2depth": "Predict the depth map image aligned with the input UAV nadir RGB.",
    "rgb2seg": "Generate the semantic segmentation map (palette PNG) for the input UAV image.",
    "seg2rgb": "Translate the segmentation map to a photorealistic RGB image.",
    "depth2rgb": "Translate the depth visualization image to a photorealistic RGB image.",
    "seg2depth": "Convert the segmentation map to a compatible depth visualization image.",
    "depth2seg": "Convert the depth visualization image to a semantic segmentation map.",
    "dseg_text2rgb": "Generate a realistic UAV RGB conditioned on depth+seg (and text if provided).",
    "rgb2rgb_recon": "Reconstruct the same RGB image faithfully.",
    "depth2depth_recon": "Reconstruct the same depth visualization image.",
    "seg2seg_recon": "Reconstruct the same semantic segmentation map (palette PNG).",
    # "xmodal": "Translate between modalities according to the instruction.",
    "d_text2rgb":   "Generate a realistic UAV RGB conditioned on depth (and text if provided).",
    "seg_text2rgb": "Generate a realistic UAV RGB conditioned on segmentation (and text if provided).",
}

# 解析 ShareGPT-i2i 风格
def parse_sharegpt_i2i(row: Dict[str, Any]) -> Tuple[List[str], List[str], str, str, Dict[str, Any]]:
    conv = row.get("conversations", [])
    user_turn, asst_turn = None, None
    for t in conv:
        if t.get("role") == "user":
            user_turn = t
        elif t.get("role") == "assistant":
            asst_turn = t

    src_images: List[str] = []
    instr = ""
    if user_turn:
        for it in user_turn.get("content", []):
            if isinstance(it, dict):
                ty = it.get("type")
                if ty == "image" and it.get("image"):
                    src_images.append(it["image"])
                elif ty == "text" and it.get("text"):
                    text = (it["text"] or "").strip()
                    if text:
                        instr = text  

    tgt_images: List[str] = []
    if asst_turn:
        for it in asst_turn.get("content", []):
            if isinstance(it, dict) and it.get("type") == "image" and it.get("image"):
                tgt_images.append(it["image"])

    meta = row.get("meta", {}) or {}
    task = str(meta.get("task") or "").strip()
    return src_images, tgt_images, instr, task, meta

def _candidate_depth_vis(scene: str, stem: str, depth_vis_root: Optional[str]) -> Optional[Path]:
    if not depth_vis_root or not scene or not stem:
        return None
    return Path(depth_vis_root) / scene / f"{stem}_depth_vis.png"

def _candidate_depth_npy(scene: str, stem: str, depth_npy_root: Optional[str], depth_npy_template: str) -> Optional[Path]:
    if not depth_npy_root or not scene or not stem:
        return None

    rel = depth_npy_template.format(scene=scene, stem=stem)
    return Path(depth_npy_root) / rel

def _maybe_rewrite_to_depth_asset(
    path: str,
    meta: Dict[str, Any],
    prefer_depth: str,
    depth_vis_root: Optional[str],
    depth_npy_root: Optional[str],
    depth_npy_template: str,
) -> str:
    """
    将给定路径在“该位置需要 depth”的前提下，尽量改写为 .npy 或 depth_vis.png。
    - 已经是 .npy 或已是 depth_vis.png：直接返回
    - prefer_depth="npy"：若存在 .npy 则返回 .npy；否则若有 vis 则返回 vis；否则保留原路径
      "vis"：若存在 vis 则 vis；否则若有 npy 则 npy；否则保留
      "keep"：直接保留
    """
    p = Path(path)
    # 已经是 .npy 或已是 depth 可视化，就不动
    if p.suffix.lower() == ".npy" or p.name.endswith("_depth_vis.png"):
        return path
    if prefer_depth == "keep":
        return path

    scene = str(meta.get("scene") or "")
    stem  = str(meta.get("stem") or "")
    cand_npy = _candidate_depth_npy(scene, stem, depth_npy_root, depth_npy_template)
    cand_vis = _candidate_depth_vis(scene, stem, depth_vis_root)

    if prefer_depth == "npy":
        if cand_npy and cand_npy.exists():
            return str(cand_npy)
        if cand_vis and cand_vis.exists():
            return str(cand_vis)
        return path
    else:  # prefer_depth == "vis"
        if cand_vis and cand_vis.exists():
            return str(cand_vis)
        if cand_npy and cand_npy.exists():
            return str(cand_npy)
        return path

def compose_image_list(task: str, src: List[str], tgt: List[str]) -> List[str]:
    """
    统一成 image_list（条件在前，监督在后；最后一个当作 target）
    """
    t = (task or "").lower()
    cond = list(src)
    goal = list(tgt)
    if t in ("rgb2rgb_recon", "depth2depth_recon", "seg2seg_recon"):
        if not goal and src:
            goal = [src[0]]
    return cond + goal

def load_jsonl_lines(jsonl_path: str) -> List[Dict[str, Any]]:
    out = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[warn] {jsonl_path}:{ln} JSON parse error: {e}")
    return out

def to_parquet(
    in_jsonls: List[str],
    out_dir: str,
    shard_size: int = 5000,
    depth_vis_root: Optional[str] = None,
    depth_npy_root: Optional[str] = None,
    depth_npy_template: str = "{scene}/{stem}_depth.npy",
    prefer_depth: str = "npy",  # "npy" | "vis" | "keep"
    fail_on_empty: bool = False,
    overwrite: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if overwrite:
        for p in glob.glob(os.path.join(out_dir, "*.parquet")):
            os.remove(p)
        pi = os.path.join(out_dir, "parquet_info.json")
        if os.path.exists(pi):
            os.remove(pi)

    schema = pa.schema([
        ("text", pa.utf8()),
        ("image_list", pa.list_(pa.utf8())),
        ("task", pa.utf8()),
    ])

    total = kept = 0
    drop_no_src = drop_no_tgt = drop_both = 0
    num_rewrite_to_npy = num_rewrite_to_vis = 0

    b_text, b_imgs, b_task = [], [], []
    shard_files: List[str] = []
    shards_meta = []

    def _flush(idx: int):
        if not b_text:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(b_text, type=pa.utf8()),
                pa.array(b_imgs, type=pa.list_(pa.utf8())),
                pa.array(b_task, type=pa.utf8()),
            ],
            schema=schema
        )
        fn = f"uav_unified_edit_{idx:05d}.parquet"
        outp = os.path.join(out_dir, fn)
        pq.write_table(table, outp, compression="zstd")
        try:
            nrg = pq.ParquetFile(outp).num_row_groups
        except Exception:
            nrg = 1
        shard_files.append(fn)
        shards_meta.append({"file": fn, "path": outp, "num_row_groups": int(nrg), "num_rows": int(table.num_rows)})
        b_text.clear(); b_imgs.clear(); b_task.clear()

    # 需要 depth 的任务：哪些位置需要替换？
    DEPTH_ON_TARGET = {"rgb2depth", "seg2depth"}
    DEPTH_ON_SOURCE = {"depth2rgb", "depth2seg", "depth2depth_recon", "dseg_text2rgb", "d_text2rgb"} 
    # DEPTH_EITHER    = {"xmodal"}  # 两侧都尝试

    idx = 0
    for jp in in_jsonls:
        items = load_jsonl_lines(jp)
        for row in items:
            total += 1
            src, tgt, instr, task, meta = parse_sharegpt_i2i(row)
            t = (task or "").lower()

            if not src and not tgt:
                drop_both += 1
                if fail_on_empty:
                    raise RuntimeError("row with empty src & tgt")
                continue
            if not src:
                drop_no_src += 1
                if fail_on_empty:
                    raise RuntimeError("row with empty src_images")
                continue
            allow_no_tgt = t in ("rgb2rgb_recon", "depth2depth_recon", "seg2seg_recon")
            if not tgt and not allow_no_tgt:
                drop_no_tgt += 1
                if fail_on_empty:
                    raise RuntimeError("row with empty tgt_images")
                continue

            def _rw_list(imgs: List[str]) -> List[str]:
                nonlocal num_rewrite_to_npy, num_rewrite_to_vis
                out = []
                for p in imgs:
                    old = p
                    newp = _maybe_rewrite_to_depth_asset(
                        p, meta, prefer_depth, depth_vis_root, depth_npy_root, depth_npy_template
                    )
                    # 统计
                    if newp != old:
                        if newp.endswith(".npy"):
                            num_rewrite_to_npy += 1
                        elif newp.endswith("_depth_vis.png"):
                            num_rewrite_to_vis += 1
                    out.append(newp)
                return out

            src_mod = list(src)
            tgt_mod = list(tgt)

            def _rw_one(p: str) -> str:
                newp = _maybe_rewrite_to_depth_asset(
                    p, meta, prefer_depth, depth_vis_root, depth_npy_root, depth_npy_template
                )
                if newp != p:
                    if newp.endswith(".npy"):
                        nonlocal num_rewrite_to_npy
                        num_rewrite_to_npy += 1
                    elif newp.endswith("_depth_vis.png"):
                        nonlocal num_rewrite_to_vis
                        num_rewrite_to_vis += 1
                return newp

            if t == "d_text2rgb":

                if len(src_mod) >= 1:
                    src_mod[0] = _rw_one(src_mod[0])

            elif t == "dseg_text2rgb":
 
                if len(src_mod) >= 1:
                    src_mod[0] = _rw_one(src_mod[0])


            elif t in {"depth2rgb", "depth2seg"}:

                for i in range(len(src_mod)):
                    src_mod[i] = _rw_one(src_mod[i])


            if t in {"rgb2depth"}:
                if len(tgt_mod) >= 1:
                    tgt_mod[-1] = _rw_one(tgt_mod[-1])

            elif t == "depth2depth_recon":

                if len(tgt_mod) >= 1:
                    tgt_mod[-1] = _rw_one(tgt_mod[-1])

            text = instr or DEFAULT_INSTR_BY_TASK.get(t, "Edit/translate the image(s).")
            images = compose_image_list(t, src_mod, tgt_mod)
            if len(images) == 0:
                drop_both += 1
                continue

            b_text.append(text)
            b_imgs.append(images)
            b_task.append(task or "")

            kept += 1
            if len(b_text) >= shard_size:
                _flush(idx); idx += 1

    _flush(idx)

    info = {
        "version": 2,
        "num_files": len(shard_files),
        "num_total_samples": sum(s["num_rows"] for s in shards_meta),
        "files": shard_files,
        "file_paths": [s["path"] for s in shards_meta],
        "shards": shards_meta,
        "stats": {
            "total": total,
            "kept": kept,
            "drop_no_src": drop_no_src,
            "drop_no_tgt": drop_no_tgt,
            "drop_both": drop_both,
            "rewrite_to_npy": num_rewrite_to_npy,
            "rewrite_to_vis": num_rewrite_to_vis,
        },
        "depth_assets": {
            "prefer_depth": prefer_depth,
            "depth_vis_root": depth_vis_root or "",
            "depth_npy_root": depth_npy_root or "",
            "depth_npy_template": depth_npy_template,
        },
    }
    with open(os.path.join(out_dir, "parquet_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[done] total={total}, kept={kept}, drop_no_src={drop_no_src}, drop_no_tgt={drop_no_tgt}, drop_both={drop_both}")
    print(f"[rewrite] ->npy={num_rewrite_to_npy}, ->vis={num_rewrite_to_vis}")
    print(f"[out] {out_dir}/parquet_info.json  (num_files={info['num_files']}, total={info['num_total_samples']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", nargs="+", required=True,
                    help="i2i/跨模态/重构 JSONL 列表（不要包含 VQA JSONL）")
    ap.add_argument("--out_dir", required=True, help="输出 Parquet 根目录")
    ap.add_argument("--shard_size", type=int, default=5000)

    # depth 资产改写相关
    ap.add_argument("--prefer_depth", choices=["npy", "vis", "keep"], default="npy",
                    help="优先改写为 .npy（默认）或 depth 可视化 PNG，或保持不改写")
    ap.add_argument("--depth_vis_root", default=None,
                    help="已有彩色 depth 可视化根目录，用于回退/或 prefer_depth=vis")
    ap.add_argument("--depth_npy_root", default=None,
                    help="16bit/float 深度 .npy 根目录")
    ap.add_argument("--depth_npy_template", default="{scene}/{stem}_depth.npy",
                    help="拼接 .npy 相对路径的模板，支持 {scene} 和 {stem}")

    ap.add_argument("--fail_on_empty", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="若存在旧分片则清空重写")
    args = ap.parse_args()

    to_parquet(
        in_jsonls=args.in_jsonl,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        depth_vis_root=args.depth_vis_root,
        depth_npy_root=args.depth_npy_root,
        depth_npy_template=args.depth_npy_template,
        prefer_depth=args.prefer_depth,
        fail_on_empty=args.fail_on_empty,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
