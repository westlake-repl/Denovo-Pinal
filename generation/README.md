# Generation Scripts

This directory contains the inference scripts and generation hyperparameters used in the paper
to design ADH, PETase, H protein, and GFP sequences.

Run all commands from the Denovo-Pinal repository root.

Before running the scripts, replace `[your_pinal_model_root]` in `generation/scripts/*.sh` with your local Pinal model directory, for example `weights/Pinal`. The directory should contain `T2struc-15B/` and `SaProt-T/`.

## Generate Sequences

```bash
bash generation/scripts/adh.sh
bash generation/scripts/petase.sh
bash generation/scripts/h_protein.sh
bash generation/scripts/gfp.sh
```

Each script runs Pinal with the generation hyperparameters used for that target and writes results to generation/outputs/<target>/.

The final gathered FASTA file is:

generation/outputs/<target>/all_seq_token.fasta

The per-rank raw outputs are moved to:

generation/outputs/<target>/rank_files/

## Targets

| Target | Description file | Script |
| --- | --- | --- |
| ADH | `generation/adh.txt` | `generation/scripts/adh.sh` |
| PETase | `generation/petase.txt` | `generation/scripts/petase.sh` |
| H protein | `generation/h_protein.txt` | `generation/scripts/h_protein.sh` |
| GFP | `generation/gfp.txt` | `generation/scripts/gfp.sh` |
