"""
Unified GearNet / ESM-GearNet embedding extractor.

Two run modes:
    --mode all           Build (accession, pdb_path, labels) from --pdb_root and
                         --train_terms_tsv, do a random 80/10/10 split, embed each
                         split with the chosen encoder.
    --mode experimental  Reuse the reference split's labels (from a prior --mode all
                         run) and a CSV manifest of downloaded experimental PDBs;
                         embed only structures present in both.

Two encoder choices:
    --encoder gearnet       Plain torchdrug.models.GearNet.
    --encoder esm_gearnet   ESM-GearNet FusionNetwork (requires --esm_weight_dir).

Per-split outputs in --output_dir:
    <split>_embeddings.pt      dict: embeddings, targets, accessions, pdb_files, split
    <split>_embeddings.npy     (optional) raw float32 matrix
    <split>_targets.npy        (optional)
    <split>_accessions.npy     (optional)
    <split>_index.csv          accession + pdb_file per row
    <split>_bad_samples.csv    only if any sample failed
    label_vocab.json
    splits.csv                 only in --mode all
"""
