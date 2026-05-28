"""Run an encoder over a DataLoader and write the resulting embeddings to disk."""

from __future__ import annotations

import gc
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchdrug import data

from .dataset import PDBDataset, collate, safe_iter
from .encoders import build_encoder
from .utils import log


def _forward_single(model: nn.Module, protein, device: str) -> torch.Tensor:
    packed = data.Protein.pack([protein])
    packed.view = "residue"
    return model(packed.to(device)).detach().cpu()


def embed_split(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    split_name: str,
    output_dir: str,
    save_pt: bool,
    save_npy: bool,
) -> None:
    model.eval()
    embs, tgts, accs, paths, bad = [], [], [], [], []

    with torch.no_grad():
        for batch in safe_iter(loader, desc=f"embed:{split_name}"):
            try:
                emb = model(batch["graph"].to(device)).detach().cpu()
                embs.append(emb)
                tgts.append(batch["target"].cpu())
                accs.extend(batch["accession"])
                paths.extend(batch["pdb_file"])
                continue
            except Exception as e:
                log(f"[{split_name}] batch failed, retrying per-sample: {e!r}")

            # Per-sample fallback so one bad protein doesn't kill the whole batch.
            for i in range(len(batch["accession"])):
                acc = batch["accession"][i]
                pdb = batch["pdb_file"][i]
                try:
                    emb_i = _forward_single(model, batch["graph"][i], device)
                    embs.append(emb_i)
                    tgts.append(batch["target"][i : i + 1].cpu())
                    accs.append(acc)
                    paths.append(pdb)
                except Exception as e_i:
                    log(f"[{split_name}] drop {acc} ({pdb}): {e_i!r}")
                    bad.append({"accession": acc, "pdb_file": pdb, "error": repr(e_i)})

    if bad:
        pd.DataFrame(bad).to_csv(
            os.path.join(output_dir, f"{split_name}_bad_samples.csv"), index=False
        )
        log(f"[{split_name}] {len(bad)} samples dropped (see *_bad_samples.csv)")

    if not embs:
        log(f"[{split_name}] no embeddings produced")
        return

    embeddings = torch.cat(embs, dim=0).contiguous()
    targets = torch.cat(tgts, dim=0).contiguous()
    log(f"[{split_name}] rows={embeddings.shape[0]} dim={embeddings.shape[1]}")

    if save_pt:
        torch.save(
            {
                "embeddings": embeddings,
                "targets": targets,
                "accessions": accs,
                "pdb_files": paths,
                "split": split_name,
            },
            os.path.join(output_dir, f"{split_name}_embeddings.pt"),
        )

    if save_npy:
        np.save(os.path.join(output_dir, f"{split_name}_embeddings.npy"), embeddings.numpy())
        np.save(os.path.join(output_dir, f"{split_name}_targets.npy"), targets.numpy())
        np.save(
            os.path.join(output_dir, f"{split_name}_accessions.npy"),
            np.asarray(accs, dtype=object),
            allow_pickle=True,
        )

    pd.DataFrame(
        {"row_idx": np.arange(len(accs)), "accession": accs, "pdb_file": paths}
    ).to_csv(os.path.join(output_dir, f"{split_name}_index.csv"), index=False)


def embed_all_splits(args, df: pd.DataFrame, splits: List[str]) -> None:
    """Build the encoder once, then loop over the requested splits."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} encoder={args.encoder}")
    model = build_encoder(args.encoder, args.checkpoint, args.esm_weight_dir).to(device)

    for s in splits:
        sub = df[df["split"] == s].reset_index(drop=True)
        if sub.empty:
            log(f"[{s}] empty split, skipping")
            continue

        ds = PDBDataset(sub, encoder_name=args.encoder, max_length=args.max_length)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )
        embed_split(
            model=model,
            loader=loader,
            device=device,
            split_name=s,
            output_dir=args.output_dir,
            save_pt=args.save_pt,
            save_npy=args.save_npy,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
