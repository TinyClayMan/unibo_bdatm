import os
import gc
import csv
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torchdrug import data, models, layers
from torchdrug.layers import geometry
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    #tf.random.set_seed(seed)

def load_split_pt(path):
    obj = torch.load(path, map_location="cpu")

    x = obj["embeddings"]
    y = obj["targets"]
    ids = obj["accessions"]

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ids = np.asarray(ids).astype(str).tolist()

    return x, y, ids

def log(msg: str):
  print(msg, flush=True)


def ensure_dir(path: str):
  os.makedirs(path, exist_ok=True)


def normalize_id(x):
  return str(x).strip()


def make_collate_fn():
  def collate(batch):
      batch = [x for x in batch if x is not None]
      if len(batch) == 0:
          return None
      proteins = [x["graph"] for x in batch]
      targets = torch.stack([x["target"] for x in batch], dim=0)
      accessions = [x["accession"] for x in batch]
      pdb_files = [x["pdb_file"] for x in batch]
      packed = data.Protein.pack(proteins)
      packed.view = "residue"
      return {
          "graph": packed,
          "target": targets,
          "accession": accessions,
          "pdb_file": pdb_files,
      }
  return collate


def safe_iter_loader(loader, desc: str, dataset_name: str, max_skip_logs: int = 20):
  skipped = 0
  it = iter(loader)
  pbar = tqdm(total=len(loader), desc=desc, leave=False)
  while True:
      try:
          batch = next(it)
          pbar.update(1)
          if batch is None:
              skipped += 1
              if skipped <= max_skip_logs:
                  log(f"[{dataset_name}] skipped empty batch")
              continue
          yield batch
      except StopIteration:
          break
      except Exception as e:
          pbar.update(1)
          skipped += 1
          if skipped <= max_skip_logs:
              log(f"[{dataset_name}] skipped failing batch during loading: {repr(e)}")
          continue
  pbar.close()
  if skipped:
      log(f"[{dataset_name}] total skipped loader batches: {skipped}")


class GearNetEmbeddingModel(nn.Module):
  def __init__(self, checkpoint_path: str):
      super().__init__()
      self.graph_construction_model = layers.GraphConstruction(
          node_layers=[geometry.AlphaCarbonNode()],
          edge_layers=[
              geometry.SequentialEdge(max_distance=2),
              geometry.SpatialEdge(radius=10.0, min_distance=5),
              geometry.KNNEdge(k=10, min_distance=5),
          ],
          edge_feature="gearnet",
      )

      self.encoder = models.GearNet(
          input_dim=21,
          hidden_dims=[512, 512, 512, 512, 512, 512],
          num_relation=7,
          edge_input_dim=59,
          num_angle_bin=8,
          batch_norm=True,
          concat_hidden=True,
          short_cut=True,
          readout="sum",
      )

      state = torch.load(checkpoint_path, map_location="cpu")
      if isinstance(state, dict):
          if "model" in state and isinstance(state["model"], dict):
              state = state["model"]
          elif "state_dict" in state and isinstance(state["state_dict"], dict):
              state = state["state_dict"]

      missing, unexpected = self.encoder.load_state_dict(state, strict=False)
      log(f"Loaded encoder checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

      for p in self.parameters():
          p.requires_grad = False

  def forward(self, packed_protein):
      packed_protein.view = "residue"
      graph = self.graph_construction_model(packed_protein)
      node_input = graph.node_feature.float()
      out = self.encoder(graph, node_input)
      return out["graph_feature"]


class ExperimentalPDBDataset(Dataset):
  def __init__(self, df: pd.DataFrame, num_labels: int, kwargs=None):
      self.df = df.reset_index(drop=True).copy()
      self.num_labels = int(num_labels)
      self.kwargs = kwargs or {}

      self.accessions = self.df["accession"].astype(str).tolist()
      self.pdb_files = self.df["pdb_file"].astype(str).tolist()
      self.targets = np.stack(self.df["target"].tolist()).astype(np.float32)

  def __len__(self):
      return len(self.df)

  def get_item(self, index):
      protein = data.Protein.from_pdb(self.pdb_files[index], **self.kwargs)
      if hasattr(protein, "residue_feature") and protein.residue_feature is not None:
          with protein.residue():
              protein.residue_feature = protein.residue_feature.to_dense()

      return {
          "graph": protein,
          "target": torch.from_numpy(self.targets[index]).float(),
          "accession": self.accessions[index],
          "pdb_file": self.pdb_files[index],
      }

  def __getitem__(self, index):
      return self.get_item(index)


def load_reference_split(split_pt):
  x, y, ids = load_split_pt(split_pt)
  ids = [normalize_id(x) for x in ids]
  return x, y, ids


def load_download_manifest(csv_path):
  df = pd.read_csv(csv_path)
  required = {"accession", "status", "download_path"}
  missing = required - set(df.columns)
  if missing:
      raise ValueError(f"Manifest missing columns: {missing}")

  df = df.copy()
  df["accession"] = df["accession"].astype(str)
  df = df[df["status"] == "downloaded"].copy()
  df = df[df["download_path"].notna()].copy()
  df = df[df["download_path"].map(os.path.exists)].copy()
  df = df[df["download_path"].astype(str).str.lower().str.endswith(".pdb")].copy()
  df = df.drop_duplicates(subset=["accession"], keep="first").reset_index(drop=True)
  return df


def build_eval_df(reference_split_pt, manifest_csv, split_name):
  _, y_ref, ids_ref = load_reference_split(reference_split_pt)
  manifest = load_download_manifest(manifest_csv)

  acc_to_row = {normalize_id(acc): i for i, acc in enumerate(ids_ref)}
  acc_to_pdb = {
      normalize_id(row["accession"]): row["download_path"]
      for _, row in manifest.iterrows()
  }

  rows = []
  missing_from_ref = 0
  for acc, pdb_path in sorted(acc_to_pdb.items()):
      i = acc_to_row.get(acc)
      if i is None:
          missing_from_ref += 1
          continue
      rows.append({
          "accession": acc,
          "pdb_file": pdb_path,
          "target": y_ref[i].astype(np.float32),
      })

  if not rows:
      raise RuntimeError(f"[{split_name}] no overlapping accessions between reference split and manifest")

  df = pd.DataFrame(rows)
  log(f"[{split_name}] downloaded structures in manifest: {len(manifest)}")
  log(f"[{split_name}] overlap with reference split: {len(df)}")
  if missing_from_ref:
      log(f"[{split_name}] manifest accessions absent from reference split: {missing_from_ref}")
  return df


def run_model_on_single(model, protein, device):
  packed = data.Protein.pack([protein])
  packed.view = "residue"
  packed = packed.to(device)
  emb = model(packed)
  return emb.detach().cpu()


def try_embed_batch(model, batch, device):
  graph = batch["graph"].to(device)
  emb = model(graph)
  return emb.detach().cpu()


def save_bad_rows(bad_rows, output_dir, split_name):
  if not bad_rows:
      return
  bad_path = os.path.join(output_dir, f"{split_name}_bad_samples.csv")
  pd.DataFrame(bad_rows).to_csv(bad_path, index=False)
  log(f"[{split_name}] wrote bad sample log to {bad_path}")


def save_split_embeddings(
  model,
  loader,
  device,
  split_name: str,
  output_dir: str,
  save_pt: bool = True,
  save_npy: bool = True,
):
  model.eval()

  all_embeddings = []
  all_targets = []
  all_accessions = []
  all_pdb_files = []
  bad_rows = []

  with torch.no_grad():
      for batch in safe_iter_loader(loader, desc=f"embed:{split_name}", dataset_name=split_name):
          try:
              emb = try_embed_batch(model, batch, device)

              all_embeddings.append(emb)
              all_targets.append(batch["target"].cpu())
              all_accessions.extend(batch["accession"])
              all_pdb_files.extend(batch["pdb_file"])
              continue

          except Exception as e:
              log(f"[{split_name}] batch forward failed, retrying sample-by-sample: {repr(e)}")

          # fallback: process one sample at a time
          batch_size = len(batch["accession"])
          recovered_embeddings = []
          recovered_targets = []
          recovered_accessions = []
          recovered_pdb_files = []

          for i in range(batch_size):
              accession = batch["accession"][i]
              pdb_file = batch["pdb_file"][i]
              target = batch["target"][i:i+1].cpu()

              try:
                  protein = batch["graph"][i]
                  emb_i = run_model_on_single(model, protein, device)

                  recovered_embeddings.append(emb_i)
                  recovered_targets.append(target)
                  recovered_accessions.append(accession)
                  recovered_pdb_files.append(pdb_file)

              except Exception as e_single:
                  log(f"[{split_name}] skipping bad sample accession={accession} pdb={pdb_file} err={repr(e_single)}")
                  bad_rows.append({
                      "accession": accession,
                      "pdb_file": pdb_file,
                      "error": repr(e_single),
                  })

          if recovered_embeddings:
              all_embeddings.append(torch.cat(recovered_embeddings, dim=0))
              all_targets.append(torch.cat(recovered_targets, dim=0))
              all_accessions.extend(recovered_accessions)
              all_pdb_files.extend(recovered_pdb_files)

  if len(all_embeddings) == 0:
      save_bad_rows(bad_rows, output_dir, split_name)
      raise RuntimeError(f"[{split_name}] no embeddings produced")

  embeddings = torch.cat(all_embeddings, dim=0).contiguous()
  targets = torch.cat(all_targets, dim=0).contiguous()

  log(f"[{split_name}] saved rows={embeddings.shape[0]} embedding_dim={embeddings.shape[1]}")
  log(f"[{split_name}] bad / skipped samples={len(bad_rows)}")

  if save_pt:
      pt_path = os.path.join(output_dir, f"{split_name}_embeddings.pt")
      torch.save(
          {
              "embeddings": embeddings,
              "targets": targets,
              "accessions": all_accessions,
              "pdb_files": all_pdb_files,
              "split": split_name,
          },
          pt_path,
      )
      log(f"[{split_name}] wrote {pt_path}")

  if save_npy:
      np.save(os.path.join(output_dir, f"{split_name}_embeddings.npy"), embeddings.numpy())
      np.save(os.path.join(output_dir, f"{split_name}_targets.npy"), targets.numpy())
      np.save(
          os.path.join(output_dir, f"{split_name}_accessions.npy"),
          np.asarray(all_accessions, dtype=object),
          allow_pickle=True,
      )
      np.save(
          os.path.join(output_dir, f"{split_name}_pdb_files.npy"),
          np.asarray(all_pdb_files, dtype=object),
          allow_pickle=True,
      )
      log(f"[{split_name}] wrote npy files")

  pd.DataFrame(
      {
          "row_idx": np.arange(len(all_accessions)),
          "accession": all_accessions,
          "pdb_file": all_pdb_files,
      }
  ).to_csv(os.path.join(output_dir, f"{split_name}_index.csv"), index=False)

  save_bad_rows(bad_rows, output_dir, split_name)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--reference_embed_dir", type=str, required=True)
  parser.add_argument("--download_root", type=str, required=True)
  parser.add_argument("--checkpoint", type=str, required=True)
  parser.add_argument("--output_dir", type=str, required=True)

  parser.add_argument("--valid_manifest", type=str, default=None)
  parser.add_argument("--test_manifest", type=str, default=None)
  parser.add_argument("--valid_pt", type=str, default=None)
  parser.add_argument("--test_pt", type=str, default=None)

  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--num_workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--save_pt", action="store_true")
  parser.add_argument("--save_npy", action="store_true")

  args = parser.parse_args()

  ensure_dir(args.output_dir)
  set_seed(args.seed)
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(args.output_dir, "torch_extensions"))

  if not args.save_pt and not args.save_npy:
      args.save_pt = True
      args.save_npy = True

  valid_manifest = args.valid_manifest or os.path.join(
      args.download_root, "valid", "valid_experimental_structures.csv"
  )
  test_manifest = args.test_manifest or os.path.join(
      args.download_root, "test", "test_experimental_structures.csv"
  )

  valid_pt = args.valid_pt or os.path.join(args.reference_embed_dir, "valid_embeddings.pt")
  test_pt = args.test_pt or os.path.join(args.reference_embed_dir, "test_embeddings.pt")

  label_vocab_path = os.path.join(args.reference_embed_dir, "label_vocab.json")
  with open(label_vocab_path) as f:
      vocab = json.load(f)
  with open(os.path.join(args.output_dir, "label_vocab.json"), "w") as f:
      json.dump(vocab, f, indent=2)

  num_labels = len(vocab["idx_to_term"])
  log(f"num_labels={num_labels}")

  valid_df = build_eval_df(valid_pt, valid_manifest, "valid")
  test_df = build_eval_df(test_pt, test_manifest, "test")

  kwargs = {
      "atom_feature": "default",
      "bond_feature": "default",
      "residue_feature": "default",
  }

  device = "cuda" if torch.cuda.is_available() else "cpu"
  log(f"device={device}")

  collate_fn = make_collate_fn()
  model = GearNetEmbeddingModel(args.checkpoint).to(device)
  model.eval()

  for split_name, sdf in [("valid", valid_df), ("test", test_df)]:
      ds = ExperimentalPDBDataset(sdf, num_labels, kwargs=kwargs)
      loader = DataLoader(
          ds,
          batch_size=args.batch_size,
          shuffle=False,
          num_workers=args.num_workers,
          collate_fn=collate_fn,
          pin_memory=torch.cuda.is_available(),
          persistent_workers=(args.num_workers > 0),
          drop_last=False,
      )

      save_split_embeddings(
          model=model,
          loader=loader,
          device=device,
          split_name=split_name,
          output_dir=args.output_dir,
          save_pt=args.save_pt,
          save_npy=args.save_npy,
      )

      gc.collect()
      if torch.cuda.is_available():
          torch.cuda.empty_cache()

  log("Done.")
  log(f"Saved experimental GearNet embeddings to: {args.output_dir}")


if __name__ == "__main__":
  main()