"""
Logging helpers: console + file, timestamps, progress with ETA, GPU memory.

Every script writes to  logs/<script>_<YYYYmmdd_HHMMSS>.log  (or --log_file) and
also keeps a symlink logs/<script>_latest.log, so during a run you can do

    tail -f logs/detect_retrieval_heads_latest.log
"""
import logging
import os
import platform
import sys
import time

LOGGER_NAME = "rh"


def get_logger():
    return logging.getLogger(LOGGER_NAME)


def add_logging_args(ap):
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--log_file", default=None, help="default: <log_dir>/<script>_<time>.log")
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                    help="DEBUG additionally logs per-layer timings, every generated answer, etc.")
    return ap


def setup_logging(script_name, args=None):
    log_dir = getattr(args, "log_dir", "logs")
    level = getattr(logging, getattr(args, "log_level", "INFO"))
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = getattr(args, "log_file", None) or os.path.join(log_dir, f"{script_name}_{stamp}.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # line-buffered stdout so `python ... | tee` / nohup show lines immediately
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    latest = os.path.join(log_dir, f"{script_name}_latest.log")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.abspath(log_file), latest)
    except OSError:
        pass

    logger.info("=" * 90)
    logger.info(f"{script_name} started | log file: {os.path.abspath(log_file)}")
    logger.info(f"command: {' '.join(sys.argv)}")
    if args is not None:
        for k, v in sorted(vars(args).items()):
            logger.info(f"  arg {k:<26s} = {v}")
    log_environment()
    return logger


def log_environment():
    logger = get_logger()
    logger.info(f"python {platform.python_version()} on {platform.platform()}")
    try:
        import torch
        logger.info(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                logger.info(f"  cuda:{i} {p.name}, {p.total_memory / 2**30:.1f} GiB")
    except ImportError:
        pass
    try:
        import transformers
        logger.info(f"transformers {transformers.__version__}")
    except ImportError:
        pass
    for pkg in ("mamba_ssm", "causal_conv1d"):
        try:
            __import__(pkg)
            logger.info(f"{pkg}: installed (fast CUDA kernels)")
        except Exception:
            logger.info(f"{pkg}: not installed (torch fallback path)")


def gpu_mem():
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        parts = []
        for i in range(torch.cuda.device_count()):
            parts.append(f"cuda:{i} {torch.cuda.memory_allocated(i) / 2**30:.1f}/"
                         f"{torch.cuda.max_memory_allocated(i) / 2**30:.1f}G")
        return " ".join(parts)
    except Exception:
        return "?"


def fmt_time(sec):
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Progress:
    """progress = Progress(total, 'detect');  ...;  msg = progress.step()  ->  '[12/120 10.0% | 0:03:10 elapsed | ETA 0:28:31]'"""

    def __init__(self, total, name=""):
        self.total = max(total, 1)
        self.name = name
        self.done = 0
        self.t0 = time.time()

    def step(self, n=1):
        self.done += n
        el = time.time() - self.t0
        eta = el / self.done * (self.total - self.done)
        return (f"[{self.name} {self.done}/{self.total} {100 * self.done / self.total:5.1f}% | "
                f"{fmt_time(el)} elapsed | ETA {fmt_time(eta)}]")


def load_model_logged(model_name, dtype):
    """Loads tokenizer + model with logging of time and memory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger = get_logger()
    t0 = time.time()
    logger.info(f"loading tokenizer {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"loading model {model_name} (dtype={dtype}, device_map=auto) ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype),
                                                 device_map="auto").eval()
    n_par = sum(p.numel() for p in model.parameters())
    logger.info(f"model loaded in {time.time() - t0:.0f}s: {n_par / 1e9:.2f}B params, "
                f"device map: {getattr(model, 'hf_device_map', {}) and sorted(set(map(str, model.hf_device_map.values())))}"
                f" | GPU mem {gpu_mem()}")
    return tok, model