"""
Unified CAFA-5 model trainer + evaluator scripts.

The files in this folder are the same scripts that were previously written
inline in notebook 03 (`cafa.py`, `train_mlp_clean_torch.py`,
`eval_experimental_vs_alphadb.py`, `eval_naive_freq_cafa.py`). They are kept
as standalone scripts so the notebook can call them with `python /content/<name>.py`
exactly as before. The notebook now starts with one cell that copies them from
this folder into `/content/`, so the `%%writefile` cells are no longer needed.

Files:
  cafa.py                          - CAFA-5 evaluator (GO graph, IA, F-max).
  train_mlp_clean_torch.py         - universal trainer for one MLP/GGN head
                                     on top of any embedding source.
  eval_experimental_vs_alphadb.py  - out-of-distribution swap test
                                     (AlphaFold PDBs vs experimental PDBs).
  eval_naive_freq_cafa.py          - naive class-frequency baseline.
"""
