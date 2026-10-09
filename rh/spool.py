"""
The files through which a run and the metrics talk. A run writes into a spool folder, the metrics read it and,
when asked, delete what they have processed; the run waits while the unread data exceeds its buffer.

    run.json                      model, layers with attention, what is saved, masked heads
    NNNNNN_<id>.sample.npz        prompt tokens, spans, reference answer, meta; written before the generation
    NNNNNN_<id>.CCCCCC.steps.npz  a chunk of consecutive generation steps
    NNNNNN_<id>.done.json         generated tokens, response text, number of chunks; whether the answer ended by itself
                                  and its text cut at every token limit of the run; written after the last chunk
    RUN_DONE                      the run has finished

A chunk holds per step and per head: the top-k attention positions and values, and the attention mass on every span
of the sample; with save="rows" also the whole attention row. Files appear under their final name only when complete.
"""
import glob
import json
import os
import time

import numpy as np

STEPS = ".steps.npz"


def write_atomic(path, write):
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def save_json(path, obj):
    def write(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    write_atomic(path, write)


def save_npz(path, **arrays):
    def write(tmp):
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
    write_atomic(path, write)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SpoolWriter:
    def __init__(self, path, save="compact", topk=5, buffer_gb=2.0, chunk_mb=256):
        assert save in ("compact", "rows", "none")
        self.path, self.save, self.topk = path, save, topk
        self.buffer = buffer_gb * 2 ** 30   # 0: no limit
        self.chunk = chunk_mb * 2 ** 20
        os.makedirs(path, exist_ok=True)
        self.pending_ids = self.clean()
        names = [os.path.basename(p) for p in glob.glob(f"{path}/*.sample.npz")]
        self.seq = max([int(n.split("_")[0]) for n in names], default=-1) + 1

    def clean(self):
        """
        Removes what an interrupted run left behind: samples without done.json and unfinished files. Returns the ids
        of the complete samples that the metrics have not read yet; they must not be generated again.
        """
        for tmp in glob.glob(f"{self.path}/*.tmp"):
            os.remove(tmp)
        pending = set()
        for sample_path in glob.glob(f"{self.path}/*.sample.npz"):
            prefix = sample_path[:-len(".sample.npz")]
            if os.path.exists(f"{prefix}.done.json"):
                pending.add(read_sample(sample_path)[1]["id"])
            else:
                print(f"removing the unfinished sample of an interrupted run: {os.path.basename(prefix)}", flush=True)
                for path in glob.glob(f"{prefix}.*"):
                    os.remove(path)
        return pending

    def start_run(self, meta):
        if os.path.exists(f"{self.path}/RUN_DONE"):
            os.remove(f"{self.path}/RUN_DONE")
        save_json(f"{self.path}/run.json", dict(meta, save=self.save, topk=self.topk))

    def start_sample(self, sample):
        self.prefix = f"{self.path}/{self.seq:06d}_{sample.id}"
        self.seq += 1
        self.span_names = sorted(sample.spans)
        self.spans = [sample.spans[n] for n in self.span_names]
        self.pending, self.pending_bytes, self.n_chunks, self.n_steps = [], 0, 0, 0
        save_npz(f"{self.prefix}.sample.npz", input_ids=np.array(sample.input_ids, dtype=np.int32),
                 info=json.dumps({"id": sample.id, "spans": sample.spans, "reference": sample.reference, "meta": sample.meta}))

    def add_step(self, token, rows):
        """rows: [layer, head, kv_len] tensor of the step, or None when attention is not saved."""
        step = {"token": token}
        if self.save != "none":
            import torch  # only the run needs it, the metrics read these files without it
            k = min(self.topk, rows.shape[-1])
            val, idx = rows.topk(k, dim=-1)
            # in half precision many positions have exactly equal attention; the top-1 is defined as the first
            # of them, topk does not guarantee it
            first = rows.argmax(dim=-1, keepdim=True)
            found = idx == first
            slot = found.float().argmax(dim=-1, keepdim=True)
            idx.scatter_(-1, slot, torch.where(found.any(-1, keepdim=True), idx[..., :1], idx.gather(-1, slot)))
            idx[..., :1] = first
            step["topk_idx"] = idx.cpu().numpy().astype(np.int32)
            step["topk_val"] = val.cpu().numpy().astype(np.float32)
            step["span_mass"] = np.stack([rows[:, :, s:e].sum(-1).cpu().numpy() for s, e in self.spans], -1).astype(np.float32)
        if self.save == "rows":
            step["row"] = rows.half().cpu().numpy()
        self.pending.append(step)
        self.pending_bytes += sum(v.nbytes for v in step.values() if isinstance(v, np.ndarray))
        self.n_steps += 1
        if self.pending_bytes >= self.chunk:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        self.wait_for_room(self.pending_bytes)
        steps = self.pending
        arrays = {"step_start": self.n_steps - len(steps), "tokens": np.array([s["token"] for s in steps], dtype=np.int32)}
        if self.save != "none":
            for key in ("topk_idx", "topk_val", "span_mass"):
                arrays[key] = np.stack([s[key] for s in steps])
        if self.save == "rows":
            for i, s in enumerate(steps):
                arrays[f"row{i}"] = s["row"]   # rows of different steps have different lengths
        save_npz(f"{self.prefix}.{self.n_chunks:06d}{STEPS}", **arrays)
        self.n_chunks += 1
        self.pending, self.pending_bytes = [], 0

    def spooled_bytes(self):
        return sum(e.stat().st_size for e in os.scandir(self.path) if e.name.endswith(STEPS))

    def wait_for_room(self, new_bytes):
        if not self.buffer:
            return
        waited = 0
        while True:
            used = self.spooled_bytes()
            if used == 0 or used + new_bytes <= self.buffer:
                return
            if waited % 60 == 0:
                print(f"spool holds {used / 2 ** 30:.2f} GB, waiting for the metrics to read it "
                      f"(python -m rh.metrics {self.path} --follow --consume)", flush=True)
            time.sleep(1)
            waited += 1

    def end_sample(self, output_ids, response, seconds, **extra):
        self.flush()
        save_json(f"{self.prefix}.done.json", {"output_ids": [int(i) for i in output_ids], "response": response,
                                                "n_chunks": self.n_chunks, "n_steps": self.n_steps, "seconds": seconds, **extra})

    def end_run(self, completed=True):
        save_json(f"{self.path}/RUN_DONE", {"completed": completed})


def read_sample(path):
    """path of a .sample.npz -> (input_ids, info dict with id, spans, reference, meta)."""
    with np.load(path) as d:
        return d["input_ids"], json.loads(str(d["info"]))


def read_steps(path):
    """Yields the steps of a chunk as dicts: step, token, and if saved topk_idx, topk_val, span_mass [layer, head, ...], row."""
    with np.load(path) as d:
        start, tokens = int(d["step_start"]), d["tokens"]
        compact = {k: d[k] for k in ("topk_idx", "topk_val", "span_mass") if k in d.files}
        for i, token in enumerate(tokens):
            step = {"step": start + i, "token": int(token)}
            step.update({k: v[i] for k, v in compact.items()})
            if f"row{i}" in d.files:
                step["row"] = d[f"row{i}"]
            yield step
