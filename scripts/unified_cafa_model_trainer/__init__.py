"""
Unified CAFA-5 model trainer + evaluator scripts.

Files:
  cafa.py                          - CAFA-5 evaluator (GO graph, IA, F-max).
  train_mlp_clean_torch.py         - universal trainer for one MLP/GGN head
                                     on top of any embedding source.
  eval_experimental_vs_alphadb.py  - out-of-distribution swap test
                                     (AlphaFold PDBs vs experimental PDBs).
  eval_naive_freq_cafa.py          - naive class-frequency baseline.
"""
