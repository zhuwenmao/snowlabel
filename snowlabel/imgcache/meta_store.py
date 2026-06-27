"""MySQL/MariaDB 元数据存储:images + masks 两张表。

存 SAM 分割出的每个 mask 的几何/bbox/PNG/parts 编码/预测,以及图片级元信息。
向量本体入 Milvus(见 vector_store);这里只存结构化元数据。连接走 SQLAlchemy
mysql+pymysql,对 MariaDB 与 MySQL 官方版通用。

线程安全:用 SQLAlchemy Engine 连接池,每次操作开独立 Session(后台预处理队列
会并发写,故不共享 Session)。
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import (BigInteger, Boolean, DateTime, Float, Integer, String,
                        Text, create_engine, func, select)
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMBLOB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

log = logging.getLogger("snowlabel.imgcache.meta")


class Base(DeclarativeBase):
    pass


class ImageRow(Base):
    """一张被处理过的图片(按内容 sha256 去重)。"""
    __tablename__ = "images"
    image_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    n_masks: Mapped[int] = mapped_column(Integer, default=0)
    minio_key: Mapped[str] = mapped_column(String(160), default="")
    orig_name: Mapped[str] = mapped_column(String(255), default="")
    # 是否已完成 SAM 分割入库。未完成(后台排队中)时 lookup 不命中。
    sam_done: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow)


class MaskRow(Base):
    """SAM 在某张图上分割出的一个 mask 及其特征/预测。"""
    __tablename__ = "masks"
    mask_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True)
    image_hash: Mapped[str] = mapped_column(String(64), index=True)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    bbox_x0: Mapped[int] = mapped_column(Integer, default=0)
    bbox_y0: Mapped[int] = mapped_column(Integer, default=0)
    bbox_x1: Mapped[int] = mapped_column(Integer, default=0)
    bbox_y1: Mapped[int] = mapped_column(Integer, default=0)
    area: Mapped[int] = mapped_column(Integer, default=0)
    # geom[4] 与 v_parts 编码成字符串(base64(float16));命中缓存时解回做重排检索。
    geom_enc: Mapped[str] = mapped_column(String(255), default="")
    whole_enc: Mapped[str] = mapped_column(Text, default="")   # v_whole base64(f16),可靠快读
    parts_enc: Mapped[str] = mapped_column(LONGTEXT, default="")
    # mask 的二值 PNG(前端直接拿来画,无需重跑 SAM)。
    mask_png: Mapped[bytes] = mapped_column(MEDIUMBLOB)
    # 首次处理时的预测(冗余存档/数据集用;命中时仍会用最新记忆库实时重判)。
    pred_label: Mapped[str] = mapped_column(String(128), default="")
    tier: Mapped[str] = mapped_column(String(16), default="silent")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow)


class MetaStore:
    """images/masks 两表的薄封装。建表幂等;每次操作独立 Session。"""

    def __init__(self, url: str):
        # pool_pre_ping 防长连接被 MySQL 掐断;pool_recycle 早于 wait_timeout。
        self.engine = create_engine(
            url, pool_pre_ping=True, pool_recycle=3600, future=True)
        Base.metadata.create_all(self.engine)
        log.info("MetaStore ready (%s)", url.split("@")[-1])

    # ---------- images ----------
    def image_done(self, image_hash: str) -> bool:
        """该图是否已 SAM 处理完入库(可命中缓存)。"""
        with Session(self.engine) as s:
            row = s.get(ImageRow, image_hash)
            return bool(row and row.sam_done)

    def get_image(self, image_hash: str) -> ImageRow | None:
        with Session(self.engine) as s:
            return s.get(ImageRow, image_hash)

    def upsert_image(self, image_hash: str, *, width: int, height: int,
                     minio_key: str = "", orig_name: str = "",
                     sam_done: bool = False, n_masks: int = 0) -> None:
        with Session(self.engine) as s:
            row = s.get(ImageRow, image_hash)
            if row is None:
                row = ImageRow(image_hash=image_hash)
                s.add(row)
            row.width, row.height = width, height
            if minio_key:
                row.minio_key = minio_key
            if orig_name:
                row.orig_name = orig_name
            row.sam_done = sam_done
            row.n_masks = n_masks
            s.commit()

    def mark_done(self, image_hash: str, n_masks: int) -> None:
        with Session(self.engine) as s:
            row = s.get(ImageRow, image_hash)
            if row is not None:
                row.sam_done = True
                row.n_masks = n_masks
                s.commit()

    # ---------- masks ----------
    def get_masks(self, image_hash: str) -> list[MaskRow]:
        with Session(self.engine) as s:
            stmt = (select(MaskRow).where(MaskRow.image_hash == image_hash)
                    .order_by(MaskRow.idx))
            return list(s.scalars(stmt))

    def replace_masks(self, image_hash: str, rows: list[dict]) -> list[int]:
        """删掉该图旧 mask(若有),写入新的一批,返回新 mask_id 列表(按 idx 顺序)。"""
        with Session(self.engine) as s:
            s.query(MaskRow).filter(MaskRow.image_hash == image_hash).delete()
            objs = [MaskRow(image_hash=image_hash, **r) for r in rows]
            s.add_all(objs)
            s.commit()
            return [o.mask_id for o in objs]

    # ---------- introspection ----------
    def stats(self) -> dict:
        with Session(self.engine) as s:
            n_img = s.scalar(select(func.count()).select_from(ImageRow)) or 0
            n_done = s.scalar(select(func.count()).select_from(ImageRow)
                              .where(ImageRow.sam_done.is_(True))) or 0
            n_mask = s.scalar(select(func.count()).select_from(MaskRow)) or 0
        return {"images": int(n_img), "images_done": int(n_done),
                "masks": int(n_mask)}
