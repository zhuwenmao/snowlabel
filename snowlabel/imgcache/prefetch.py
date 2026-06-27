"""后台预处理队列:把上传的图片排队,后台逐张跑 SAM 入库(预取)。

这样用户标注当前图时,后续图可能已被后台 SAM 分割好并写入三库,翻页即命中缓存
秒出。worker 调 predictor.segment_all_cached(未命中才真算),与前台共用 GPU 锁串行;
每张之间小睡,让前台交互优先抢锁。单 worker、守护线程、失败不中断、同图不重复入队。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from PIL import Image

log = logging.getLogger("snowlabel.imgcache.prefetch")


class Prefetcher:
    def __init__(self, predictor, max_masks: int = 16, gap_sec: float = 0.4):
        self.predictor = predictor
        self.max_masks = max_masks
        self.gap_sec = gap_sec
        self._q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._seen: set[tuple[str, str]] = set()
        self._seen_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="snow-prefetch")
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()
            log.info("prefetch worker started")

    def enqueue(self, path: str, user_id: str = "") -> None:
        key = (str(path), user_id or "")
        with self._seen_lock:
            if key in self._seen:
                return
            self._seen.add(key)
        self._q.put(key)

    def pending(self) -> int:
        return self._q.qsize()

    def _run(self) -> None:
        while True:
            path_s, user_id = self._q.get()
            try:
                p = Path(path_s)
                if p.exists():
                    data = p.read_bytes()
                    img = Image.open(p).convert("RGB")
                    _, hit = self.predictor.segment_all_cached(
                        data, img, max_masks=self.max_masks,
                        user_id=user_id or None,
                        ext=(p.suffix or ".jpg"), orig_name=p.name)
                    log.info("prefetch %s %s", p.name,
                             "HIT(skip)" if hit else "processed")
            except Exception as e:
                log.warning("prefetch failed %s: %s", path_s, e)
            finally:
                self._q.task_done()
                time.sleep(self.gap_sec)  # 让前台交互优先抢 GPU 锁


_prefetcher: Prefetcher | None = None


def get_prefetcher(predictor) -> Prefetcher:
    """进程内单例 worker;首次调用时启动。"""
    global _prefetcher
    if _prefetcher is None:
        _prefetcher = Prefetcher(predictor)
        _prefetcher.start()
    return _prefetcher
