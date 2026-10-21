"""MinIO 对象存储:存图片本体。按内容 sha256 命名,同一张图只存一份(天然去重)。"""
from __future__ import annotations

import io
import logging

from minio import Minio

log = logging.getLogger("snowlabel.imgcache.object")


class ObjectStore:
    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 bucket: str, secure: bool = False):
        self.client = Minio(endpoint, access_key=access_key,
                            secret_key=secret_key, secure=secure)
        self.bucket = bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
        log.info("ObjectStore ready (minio %s / bucket %s)", endpoint, bucket)

    @staticmethod
    def key_for(image_hash: str, ext: str = ".jpg") -> str:
        return f"images/{image_hash}{ext}"

    def put_image(self, image_hash: str, data: bytes, ext: str = ".jpg",
                  content_type: str = "image/jpeg") -> str:
        """写入图片(若已存在则覆盖,内容相同故无副作用),返回对象 key。"""
        key = self.key_for(image_hash, ext)
        self.client.put_object(self.bucket, key, io.BytesIO(data),
                               length=len(data), content_type=content_type)
        return key

    def put_bytes(self, key: str, data: bytes,
                  content_type: str = "application/octet-stream") -> str:
        self.client.put_object(self.bucket, key, io.BytesIO(data),
                               length=len(data), content_type=content_type)
        return key

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except Exception:
            return False

    def get(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def delete(self, key: str) -> None:
        try:
            self.client.remove_object(self.bucket, key)
        except Exception:
            log.debug("remove_object ignored for %s", key, exc_info=True)
