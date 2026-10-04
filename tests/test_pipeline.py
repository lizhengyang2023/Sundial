from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch

from sundial.data import (BalancedCorpus, SundialIterableDataset, convert_source,
                          discover_files, extract_univariate)
from sundial.model import Sundial, SundialConfig


def test_three_source_s3_and_one_training_update(tmp_path: Path):
    source_data = tmp_path / "raw"
    corpus_root = tmp_path / "corpus"
    for name in ("utsd", "chronos", "lotsa"):
        folder = source_data / name
        folder.mkdir(parents=True)
        t = np.arange(160, dtype=np.float32)
        table = pa.Table.from_pylist([
            {"id": "one", "target": (t + 0.1).tolist()},
            {"id": "two", "target": np.sin(t / 8).tolist()},
        ])
        if name == "utsd":
            tier = folder / "UTSD-1G"
            tier.mkdir()
            with ipc.new_stream(tier / "toy.arrow", table.schema) as writer:
                writer.write_table(table)
        else:
            domain = folder / "domain"
            domain.mkdir()
            pq.write_table(table, domain / "toy.parquet")
            if name == "lotsa":
                (domain / "dataset_info.json").write_text("{}", encoding="utf-8")
        pq.write_table(table, folder / "ETTh1.parquet")
        assert len(list(discover_files(folder))) == 2
        manifest = convert_source(name, folder, corpus_root, shard_rows=1, read_batch_size=1)
        assert manifest["train_series"] == 2
        assert manifest["skipped"]["evaluation"] == 1
    cfg = SundialConfig(patch_size=16, max_context=64, horizon=32, layers=2,
                        dim=32, ff_dim=64, heads=4, flow_dim=32, flow_layers=2,
                        flow_steps=3, dropout=0)
    sampler = BalancedCorpus(corpus_root, cfg.patch_size, cfg.max_context, cfg.horizon)
    batch = sampler.sample(3, 0)
    assert batch["context"].shape[0] == 3
    iterable = SundialIterableDataset(sampler, batch_size=3, steps_per_epoch=3, seed=11)
    iterable.set_epoch(2)
    expected = list(iter(iterable))
    iterable.set_epoch(2, start_step=1)
    resumed = list(iter(iterable))
    assert len(expected) == 3 and len(resumed) == 2
    assert all(torch.equal(a["context"], b["context"])
               for a, b in zip(expected[1:], resumed))
    model = Sundial(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.flow.output.weight.detach().clone()
    loss = model.training_loss(**batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.flow.output.weight.detach())
    model.eval()
    forecast = model.forecast(batch["context"], 20, samples=2,
                              valid=batch["context_valid"])
    assert forecast.shape == (3, 2, 20)
    assert torch.isfinite(forecast).all()


def test_extract_univariate_ignores_numeric_timestamps():
    row = {"timestamp": np.arange(4),
           "time_idx": np.arange(4),
           "target_name": np.arange(4),
           "target": np.arange(8).reshape(2, 4),
           "other": np.arange(4)}
    series = list(extract_univariate(row))
    assert [name for name, _ in series] == ["target:0", "target:1", "other"]
    assert all(len(values) == 4 for _, values in series)


def test_decoder_is_causal():
    cfg = SundialConfig(patch_size=16, max_context=64, horizon=32, layers=2,
                        dim=32, ff_dim=64, heads=4, flow_dim=32, flow_layers=2,
                        dropout=0)
    model = Sundial(cfg).eval()
    x = torch.randn(1, 64)
    valid = torch.ones_like(x, dtype=torch.bool)
    altered = x.clone()
    altered[:, -16:] += 1000
    h1 = model.encode(x, valid)[0]
    h2 = model.encode(altered, valid)[0]
    # ReNorm changes when the full context changes; isolate Transformer causality
    # by testing equal normalized token inputs through its decoder blocks.
    tokens = torch.randn(1, 4, cfg.dim)
    changed = tokens.clone()
    changed[:, -1] += 1000
    mask = torch.ones(1, 4, dtype=torch.bool)
    for block in model.blocks:
        tokens = block(tokens, mask)
        changed = block(changed, mask)
    torch.testing.assert_close(tokens[:, :-1], changed[:, :-1], atol=1e-5, rtol=1e-5)
    assert h1.shape == h2.shape == (1, 4, cfg.dim)
