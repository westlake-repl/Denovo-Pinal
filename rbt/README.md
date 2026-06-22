# For 3.1 PETases analysis

Run all commands from the Denovo-Pinal repository root. Before running the design scripts, replace `[your_pinal_model_root]` in `rbt/scripts/*.sh` with the local Pinal model directory, for example `weights/Pinal`. The directory should contain `T2struc-15B/` and `SaProt-T/`.

## Generate designed PETase sequences

```bash
bash rbt/scripts/petase.sh
```

This script generates PETase designs and gathers the outputs into all_seq_token.fasta.

## Recalculate NLL for pinalpetase36 and pinalpetase_old

See `rbt/scripts/nll_cal.ipynb`

The notebook recalculates the NLL/PPL values and compares them against the designed PETase distribution.

## MMseqs search: designed PETase vs. pinalpetase36 and pinalpetase_old

Before running, make sure the MMSEQS path in rbt/search/mmseqs.sh points to a valid MMseqs binary.

`bash rbt/search/mmseqs.sh`

# For 3.2 ADH/H-protein analysis

> Before running these scripts, make sure you have an environment that can run [ProTrek](https://github.com/westlake-repl/ProTrek), or the [Pinal training](https://github.com/SuperCarryDFY/b097bade29ef6c984ea1c6f28ec5ece165c3765a459f16085d618a97f9a60286) environment that includes the ProTrek scoring code. Then replace the paths in rbt/scripts/cal_protrek_score.py (line7, line58-61) with your local ProTrek model and code paths. Also make sure you have an environment with ESMFold and its dependencies installed.

Generate designed ADH and H-protein sequences:

```bash
bash rbt/scripts/adh.sh
bash rbt/scripts/h_protein.sh
```

Calculate ProTrek scores:

```bash
python rbt/scripts/cal_protrek_score.py rbt/outputs/adh
python rbt/scripts/cal_protrek_score.py rbt/outputs/h_protein
```

Run ESMFold:

```bash
bash rbt/scripts/run_esmfold.sh rbt/outputs/adh/all_seq_token.fasta
bash rbt/scripts/run_esmfold.sh rbt/outputs/h_protein/all_seq_token.fasta
```

See `rbt/scripts/ppl_analysis.ipynb` for the analysis of ProTrek score, ESMFold pLDDT, and PPL.
