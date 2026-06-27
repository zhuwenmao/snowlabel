"""图片处理缓存(三库)。

把"一张图 → SAM mask → DINO 向量"的昂贵计算结果持久化,同图再来直接读库、
跳过 SAM/DINO。三库分工:
  - MinIO   : 图片本体(对象存储,按内容哈希命名,天然去重)      —— object_store.py
  - MySQL   : 每个 mask 的几何/bbox/PNG/parts/预测 等结构化元数据 —— meta_store.py
  - Milvus  : 每个 mask 的 DINO v_whole 向量(可被 ANN 检索)        —— vector_store.py
ImageCache(cache.py) 是编排层,对外只暴露 lookup / store。
"""
from .cache import ImageCache, make_image_cache, CachedMask

__all__ = ["ImageCache", "make_image_cache", "CachedMask"]
