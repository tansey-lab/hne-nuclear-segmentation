"""Progress + logging helpers."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from tqdm import tqdm

log = logging.getLogger("hne")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once. Safe to call multiple times."""
    root = logging.getLogger()
    if getattr(setup_logging, "_done", False):
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level)
    setup_logging._done = True  # type: ignore[attr-defined]


def attach_vips_progress(image, desc: str):
    """Wire a tqdm progress bar to a pyvips image's eval signal.

    Returns the image (for chaining). The bar closes on posteval.
    """
    pbar = tqdm(total=100, desc=desc, unit="%", leave=False)
    state = {"last": 0.0}

    def eval_cb(_img, progress):
        now = time.time()
        if (now - state["last"]) > 0.25:
            pbar.n = int(progress.percent)
            pbar.refresh()
            state["last"] = now

    def posteval_cb(_img, _progress):
        pbar.n = 100
        pbar.refresh()
        pbar.close()

    image.set_progress(True)
    image.signal_connect("eval", eval_cb)
    image.signal_connect("posteval", posteval_cb)
    return image


@contextmanager
def step(name: str):
    """Log start/end of a pipeline step with elapsed time."""
    log.info("▶ %s", name)
    t0 = time.time()
    try:
        yield
    finally:
        log.info("✓ %s (%.2fs)", name, time.time() - t0)
