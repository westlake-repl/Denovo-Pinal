# For 3.1 PETases analysis

Run all commands from the Denovo-Pinal repository root.

## Generate designed PETase sequences

```bash
bash rbt/PETase/scripts/petase.sh
```

This script generates PETase designs and gathers the outputs into sequence.fasta.

## Recalculate NLL for pinalpetase36 and pinalpetase_old

See:

rbt/PETase/scripts/nll_cal.ipynb

The notebook recalculates the NLL/PPL values and compares them against the designed PETase
distribution.

## MMseqs search: designed PETase vs. pinalpetase36 and pinalpetase_old

Before running, make sure the MMSEQS path in rbt/PETase/search/mmseqs.sh points to a valid
MMseqs binary.

bash rbt/PETase/search/mmseqs.sh

# For 3.2 ADH/H-protein analysis