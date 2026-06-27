"""Milvus 缓存集合:存每个 mask 的 v_whole 向量。

定位:向量的持久档案 + 可做跨图 ANN 相似检索(数据集价值)。命中缓存的即时读取走
MySQL 的 whole_enc(规避 milvus-lite 写后读的秒级延迟,与记忆库 RAM 镜像同理);
这里专注把向量可靠落进 Milvus。用独立 lite 文件,不与记忆库 collection 混。
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("snowlabel.imgcache.vector")


class MaskVectorStore:
    def __init__(self, collection: str, dim: int, uri: str = "", token: str = ""):
        from pymilvus import DataType, MilvusClient
        self._DataType = DataType
        self.collection = collection
        self.dim = dim
        if not uri:
            from ..config import get_config
            uri = str(get_config().memory_dir / "snowlabel_imgcache.db")
        self._client = MilvusClient(uri=uri, token=token or "")
        self._ensure_collection()
        log.info("MaskVectorStore ready (collection=%s, dim=%d)", collection, dim)

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self.collection):
            self._client.load_collection(self.collection)
            return
        schema = self._client.create_schema(auto_id=False,
                                            enable_dynamic_field=False)
        schema.add_field("mask_id", self._DataType.INT64, is_primary=True)
        schema.add_field("image_hash", self._DataType.VARCHAR, max_length=64)
        schema.add_field("vector", self._DataType.FLOAT_VECTOR, dim=self.dim)
        self._client.create_collection(self.collection, schema=schema,
                                       index_params=None)
        idx = self._client.prepare_index_params()
        idx.add_index(field_name="vector", index_type="FLAT",
                      metric_type="COSINE", index_name="vec_idx")
        self._client.create_index(self.collection, idx)
        self._client.load_collection(self.collection)

    def add(self, mask_ids, image_hash: str, vectors) -> None:
        rows = [{"mask_id": int(mid), "image_hash": image_hash,
                 "vector": np.asarray(v, dtype=np.float32).tolist()}
                for mid, v in zip(mask_ids, vectors)]
        if rows:
            self._client.insert(self.collection, rows)

    def query_by_image(self, image_hash: str):
        """按图取该图所有 mask 的 (mask_id, vector)。milvus-lite 写后可能有延迟,
        故缓存读路径不依赖它;仅供诊断 / 未来 ANN 扩展。"""
        try:
            res = self._client.query(
                self.collection, filter=f'image_hash == "{image_hash}"',
                output_fields=["mask_id", "vector"], limit=4096)
            return [(int(r["mask_id"]), np.asarray(r["vector"], dtype=np.float32))
                    for r in res]
        except Exception:
            return []

    def count(self) -> int:
        try:
            r = self._client.get_collection_stats(self.collection)
            return int(r.get("row_count", 0))
        except Exception:
            return 0
