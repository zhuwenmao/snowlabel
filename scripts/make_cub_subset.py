#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 CUB-200-2011 全量 tgz 里抽一个小而难的细粒度子集。

为什么选这些类：海鸥(Gull)/燕鸥(Tern)/鹪鹩(Wren) 三组，组内各物种长得几乎
一样（白灰海鸟、小褐鸟），是细粒度识别里公认最难分的，最能展示 SnowLabel
"教几张就越标越准"的卖点。默认 22 类、约 1300 张图，够小够快。

直接从 .tgz 流式读取，不整包解压（全量解压要 1GB+ 临时空间）。保留原始
image_id / class_id 以便溯源回官方 CUB；只是把行过滤成子集。

用法（服务器无关，纯离线）：
    python scripts/make_cub_subset.py \
        --tgz data/datasets/CUB_200_2011.tgz \
        --out data/datasets/cub200_subset \
        --keywords Gull Tern Wren
可选 --max-per-class N 进一步压缩每类图数（默认不限，保留官方 train/test 划分）。
"""
import argparse
import io
import os
import tarfile
from collections import defaultdict

ROOT = "CUB_200_2011"  # tgz 内顶层目录名
META_FILES = [
    "classes.txt",
    "images.txt",
    "image_class_labels.txt",
    "train_test_split.txt",
    "bounding_boxes.txt",
]


def _read_member_lines(tf, name):
    """读 tgz 内一个文本文件，返回去掉行尾的行列表。"""
    m = tf.getmember(f"{ROOT}/{name}")
    data = tf.extractfile(m).read().decode("utf-8", "replace")
    return [ln.rstrip("\n").rstrip("\r") for ln in data.splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgz", default="data/datasets/CUB_200_2011.tgz")
    ap.add_argument("--out", default="data/datasets/cub200_subset")
    ap.add_argument("--keywords", nargs="+", default=["Gull", "Tern", "Wren"],
                    help="类名包含任一关键词即入选（大小写不敏感）")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="每类最多保留多少张（0=不限，保留官方全量）")
    args = ap.parse_args()

    if not os.path.exists(args.tgz):
        raise SystemExit(f"找不到 tgz：{args.tgz}（先把全量包下完）")

    kws = [k.lower() for k in args.keywords]
    img_out = os.path.join(args.out, "images")
    os.makedirs(img_out, exist_ok=True)

    print(f"打开 {args.tgz} …")
    tf = tarfile.open(args.tgz, "r:gz")

    # 1) 元数据全读进内存（都是小文本）
    classes = _read_member_lines(tf, "classes.txt")          # "1 001.Black_footed_Albatross"
    images = _read_member_lines(tf, "images.txt")            # "1 001.xxx/yyy.jpg"
    labels = _read_member_lines(tf, "image_class_labels.txt")  # "1 1"
    splits = _read_member_lines(tf, "train_test_split.txt")  # "1 0"
    bboxes = _read_member_lines(tf, "bounding_boxes.txt")    # "1 60 27 325 304"

    cid2name = {}
    for ln in classes:
        cid, name = ln.split(" ", 1)
        cid2name[cid] = name

    # 2) 关键词选类。按词匹配而非子串——否则 "tern" 会误命中 Eas-tern/Wes-tern。
    #    类名形如 "141.Artic_Tern"，去掉 "NNN." 前缀后按下划线分词整词比对。
    def _tokens(name):
        species = name.split(".", 1)[-1]          # "Artic_Tern"
        return {t.lower() for t in species.split("_")}
    sel_cids = [cid for cid, name in cid2name.items()
                if _tokens(name) & set(kws)]
    sel_cids_set = set(sel_cids)
    if not sel_cids:
        raise SystemExit(f"没有类名匹配关键词 {args.keywords}")
    print(f"入选 {len(sel_cids)} 类：")
    for cid in sorted(sel_cids, key=lambda x: int(x)):
        print(f"  {cid}  {cid2name[cid].split('.',1)[-1]}")

    # 3) 选图：属于入选类的 image_id
    img2cls = {}
    for ln in labels:
        iid, cid = ln.split()
        img2cls[iid] = cid
    iid2path = {}
    for ln in images:
        iid, path = ln.split(" ", 1)
        iid2path[iid] = path

    # 每类候选 image_id（按 image_id 排序，便于 max-per-class 截断可复现）
    cls2iids = defaultdict(list)
    for iid, cid in img2cls.items():
        if cid in sel_cids_set:
            cls2iids[cid].append(iid)
    sel_iids = []
    for cid, iids in cls2iids.items():
        iids = sorted(iids, key=lambda x: int(x))
        if args.max_per_class and len(iids) > args.max_per_class:
            iids = iids[:args.max_per_class]
        sel_iids.extend(iids)
    sel_iids_set = set(sel_iids)
    print(f"入选 {len(sel_iids)} 张图。开始抽取…")

    # 4) 流式抽图。遍历整个 tar 一次，命中就写出（比逐个 getmember 快得多）
    n_written = 0
    prefix = f"{ROOT}/images/"
    # 反查：path -> iid，用于判断该成员是否入选
    path2iid = {p: i for i, p in iid2path.items()}
    for m in tf:
        if not m.isfile() or not m.name.startswith(prefix):
            continue
        rel = m.name[len(prefix):]            # "001.xxx/yyy.jpg"
        iid = path2iid.get(rel)
        if iid is None or iid not in sel_iids_set:
            continue
        dst = os.path.join(img_out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(tf.extractfile(m).read())
        n_written += 1
        if n_written % 200 == 0:
            print(f"  …已写 {n_written}/{len(sel_iids)}")
    print(f"图片写出完成：{n_written} 张 → {img_out}")

    # 5) 写过滤后的元数据（保留原始 id，便于溯源）
    def _filt_by_iid(lines):
        return [ln for ln in lines if ln.split()[0] in sel_iids_set]

    out_meta = {
        "classes.txt": [ln for ln in classes if ln.split(" ", 1)[0] in sel_cids_set],
        "images.txt": _filt_by_iid(images),
        "image_class_labels.txt": _filt_by_iid(labels),
        "train_test_split.txt": _filt_by_iid(splits),
        "bounding_boxes.txt": _filt_by_iid(bboxes),
    }
    for fn, lines in out_meta.items():
        with open(os.path.join(args.out, fn), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(f"元数据写出完成：{', '.join(out_meta)} → {args.out}")

    # 6) README
    n_train = sum(1 for ln in out_meta["train_test_split.txt"] if ln.split()[1] == "1")
    n_test = len(sel_iids) - n_train
    readme = f"""# CUB-200-2011 细粒度子集（SnowLabel demo 用）

从官方 **CUB-200-2011**（Caltech-UCSD Birds, 200 类鸟）抽取的小子集。

- **来源**：Wah et al., 2011, Caltech-UCSD Birds-200-2011（CNS-TR-2011-001）。
  下载镜像 fast.ai：`s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz`。
- **选类逻辑**：类名含关键词 {args.keywords} —— 海鸥/燕鸥/鹪鹩三组，组内物种
  外观高度相似，是细粒度识别公认的难点，最能体现 SnowLabel "教几张 mask 就
  越标越准" 的检索式自动标注效果。
- **规模**：{len(sel_cids)} 类 / {len(sel_iids)} 张图（官方划分 train {n_train} / test {n_test}）。
- **结构**：
  - `images/<类名>/*.jpg`  —— 按类分文件夹，直接可喂给标注器。
  - `classes.txt` `images.txt` `image_class_labels.txt` `train_test_split.txt`
    `bounding_boxes.txt` —— 与官方同格式，**保留原始 image_id / class_id**，
    可溯源回全量 CUB。
- **复现**：`python scripts/make_cub_subset.py --keywords {' '.join(args.keywords)}`

> 仅用于学习/演示。版权归原作者，商用请遵循 CUB-200-2011 原始许可。
"""
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    tf.close()

    # 7) 汇总
    size_mb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(args.out) for fn in fns
    ) / 1e6
    print("\n========== 子集就绪 ==========")
    print(f"  目录   : {args.out}")
    print(f"  类数   : {len(sel_cids)}")
    print(f"  图片   : {len(sel_iids)}  (train {n_train} / test {n_test})")
    print(f"  占用   : {size_mb:.1f} MB")
    print("==============================")


if __name__ == "__main__":
    main()
