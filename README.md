## Pinal: Toward De Novo Protein Design from Natural Language

<a href="https://www.biorxiv.org/content/10.1101/2024.08.01.606258"><img src="https://img.shields.io/badge/Paper-bioRxiv-green" style="max-width: 100%;"></a>
<a href="http://www.denovo-pinal.com/"><img src="https://img.shields.io/badge/Pinal-red?label=Server" style="max-width: 100%;"></a>
<a href="https://huggingface.co/westlake-repl/Pinal"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yellow?label=Model" style="max-width: 100%;"></a>
<a href="https://x.com/duguyuan/status/1877623852299096198"><img src="https://img.shields.io/badge/X-black?label=Post" style="max-width: 100%;"></a>

The repository is an official implementation of Pinal: [Toward De Novo Protein Design from Natural Language](https://www.biorxiv.org/content/10.1101/2024.08.01.606258v7)

Quickly try our online server (Pinal 16B) [here](http://www.denovo-pinal.com/)

Also try [SaProt-T/SaProt-O](http://113.45.254.183:9527/) for protein redesign/editing

If you have any questions about the paper or the code, feel free to raise an issue!

> We have 2 PhD positions for international students at Westlake University, China! see [here](https://x.com/duguyuan/status/1897101692665258245).  

### Environment setup

Create and activate a new conda environment with Python 3.8.
```shell
conda create -n pinal python=3.8 --yes
conda activate pinal
pip install -r requirements.txt
```
> It takes about 30 minutes to install the required packages.

### Download model weights

We provide a script to download the pre-trained model weights, as shown below. Please download all files and put them in the `weights` directory, e.g., `weights/Pinal/...`


```shell
huggingface-cli download westlake-repl/Pinal \
                         --repo-type model \
                         --local-dir weights/
```

#### Model checkpoints

The `weights` directory contains 4 models:

|**Name** |**Size** |
|---------|---------|
|[SaProt-T](https://huggingface.co/westlake-repl/Pinal/tree/main/SaProt-T) | 760M |
|[SaProt-O](https://huggingface.co/westlake-repl/Pinal/tree/main/SaProt-O) | 760M |
|[T2struc-1.2B](https://huggingface.co/westlake-repl/Pinal/tree/main/T2struc-1.2B) | 1.2B |
|[T2struc-15B](https://huggingface.co/westlake-repl/Pinal/tree/main/T2struc-15B) | 15.5B |


### Inference with Pinal

Design protein from natural language instruction with only 3 lines of code!

```python
from utils.design_utils import load_pinal, PinalDesign
load_pinal()
res = PinalDesign(desc="Actin.", num=10)
# res is a list of designed proteins, sorted by the probability per token. 
```

The above code will generate 10 de novo designed proteins based on the input description "Actin.", inferred by 1.2B T2struc and SaProt-T. If you want inference with T2struc-15B, you can set the environment variable `T2struc_NAME` before calling `load_pinal()`, as shown below.

```python
import os
os.environ["T2struc_NAME"] = "T2struc-15B"
```
> Warning: Inferencing with T2struc-15B requires at least 40GB GPU memory. On a single NVIDIA A40 GPU, designing 10 proteins takes approximately 1 minute. 

### Predicting amino acid sequence with SaProt-T

Here, we provide a script for predicting amino acid sequences using natural language, enabling you to specify the desired structure.

```python
from utils.design_utils import SaProtPrepareGenerationInputs, SaProtGeneration, load_SaProtT_and_tokenizers
desc = "Actin."
saprot, saprot_text_tokenizer, saprot_tokenizer = load_SaProtT_and_tokenizers()
structure = "dqdppafakewedfqfwifidtfpdqggqdifgqkkwafpdpppcvppdddridgtvrrvvvvvgtdmdgqdalqagpdpvsvlvvvvcvdcprvnhqqlnheyeyegaapydlvrllsvvccscpvsvhqwyayaylqlllcvlvvdqfawefaaalqwtkiwggdnsdtdnqlididrdhnvlllvllqvvvvvvvdhqddpnssvvssvcqlpqaaadldlvvqvvclvvdqpskdwdqdpvrdididtssrhvslccqcvvvsvvdpdhhslvsnvsslvsddpvrslvhqchyeyaysrvqhhcpqsnsqvsncvvddvphdgdydydnvrncssvssvsplspdpvnpvlidgsvncvvppssvnvvrhd"
SaProtInputDict = SaProtPrepareGenerationInputs([" ".join(list(structure))], desc, saprot_text_tokenizer, saprot_tokenizer)
seq = SaProtGeneration(saprot, SaProtInputDict, saprot_tokenizer)["sequence"]
print(seq)
```

The above code makes predictions based on Foldseek tokens. If you want to convert a 3D structure file (e.g., .pdb or .mmcif) into Foldseek tokens, you should download the binary file from [here](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view) and place it in the assets/bin folder. The following code demonstrates how to use it.
```python
from utils.foldseek_utils import get_struc_seq
pdb_path = "assets/8ac8.cif"
# Extract the "A" chain from the pdb file and encode it into a struc_seq
foldseek_seq = get_struc_seq("assets/bin/foldseek", pdb_path, ["A"])["A"][1].lower()
print(f"foldseek_seq: {foldseek_seq}")
# foldseek_seq: dfqka...ggvvd
```


### Computational evaluation of the de novo designed proteins

For textual alignment, we recommend using [ProTrek](https://github.com/westlake-repl/ProTrek) to calculate the sequence-text similarity score.

For foldability, we recommend using pLDDT and PAE, outputted by [Alphafold series](https://golgi.sandbox.google.com/) or [ESMFold](https://github.com/facebookresearch/esm).


### Other resources

- [ProTrek](https://www.biorxiv.org/content/10.1101/2024.05.30.596740v2) and its [online server](http://search-protrek.com/)
- [Evolla](https://www.biorxiv.org/content/10.1101/2025.01.05.630192v1) and its [online server](http://www.chat-protein.com/)
- [SaprotHub](https://www.biorxiv.org/content/10.1101/2024.05.24.595648v5) and its [online server](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/SaprotHub_v2.ipynb?hl=en)

