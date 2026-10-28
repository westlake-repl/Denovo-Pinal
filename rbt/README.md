# Analysis for Question 3.1: PETase

Run all shell commands from the root directory of the Denovo-Pinal repository. The notebooks in `rbt/scripts/analysis` should be run from their own directory. Before running the design scripts, replace `[your_pinal_model_root]` in `rbt/scripts/design/*.sh` with your local Pinal model directory, for example `weights/Pinal`. This directory should contain `T2struct-15.5B/` and `SaProt-T/`.

## Generate PETase sequence pool

```bash
bash rbt/scripts/design/petase.sh
```

This script generates PETase designs and gathers the generated sequences into `sequence.fasta`.
We also provide the pre-generated PETase design results produced by this script in `rbt/outputs/petase`.

## Recalculate NLL for Experimentally Validated PETases

See `rbt/scripts/analysis/nll_cal.ipynb`

The notebook recalculates the NLL/PPL values and compares them against the designed PETase distribution.

## MMseqs Search: Designed PETase vs. Experimentally Validated PETases

Before running, make sure the MMseqs2 executable path in `rbt/search/mmseqs.sh` points to a valid MMseqs2 binary.

This script searches the designed PETase sequence pool against the experimentally validated PETase sequences in `rbt/search/wetlab_seqs.fasta`, including `pinalpetase36` and the original variants `pinalpetase01`, `pinalpetase02`, `pinalpetase03`, and `pinalpetase07`.

```bash
bash rbt/search/mmseqs.sh
```

We provide the precomputed MMseqs2 search results in `rbt/search/wetlab_seqs_mmseqs_hit.tsv`.

## MMseqs2 Pairwise Search within a 1,000-sequence PETase Pool

See `rbt/scripts/analysis/petase_selfsearch.ipynb` for the pairwise sequence-similarity self-search and clustering analysis of a randomly sampled 1,000-sequence PETase design pool.

# Analysis for Question 3.2: ADH and H-protein

Before running these scripts, make sure you have an environment that can run [ProTrek](https://github.com/westlake-repl/ProTrek), or use the [Pinal training](https://github.com/SuperCarryDFY/b097bade29ef6c984ea1c6f28ec5ece165c3765a459f16085d618a97f9a60286) environment that includes the ProTrek scoring code. Then replace the paths in `rbt/scripts/eval/cal_protrek_score.py` (line7, line58-61) with your local ProTrek model and code paths. You also need an environment with ESMFold and its dependencies installed.

## Generate ADH and H-protein sequence pool

```bash
bash rbt/scripts/design/adh.sh
bash rbt/scripts/design/h_protein.sh
```

The pre-generated ADH and H-protein design results produced by these scripts are also provided in `rbt/outputs/adh` and `rbt/outputs/h_protein`, so rerunning the large-scale generation step is optional.

## Compute ProTrek Scores

```bash
python rbt/scripts/eval/cal_protrek_score.py rbt/outputs/adh
python rbt/scripts/eval/cal_protrek_score.py rbt/outputs/h_protein
```

## Run ESMFold

```bash
bash rbt/scripts/eval/run_esmfold.sh rbt/outputs/adh/sequence.fasta
bash rbt/scripts/eval/run_esmfold.sh rbt/outputs/h_protein/sequence.fasta
```

See `rbt/scripts/analysis/ppl_analysis.ipynb` for the analysis of the relationship among ProTrek score, ESMFold pLDDT, and PPL.
