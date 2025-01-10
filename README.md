## Pinal: Toward De Novo Protein Design from Natural Language

<a href="https://www.biorxiv.org/content/10.1101/2024.08.01.606258"><img src="https://img.shields.io/badge/Paper-bioRxiv-green" style="max-width: 100%;"></a>
<a href="http://www.denovo-pinal.com/"><img src="https://img.shields.io/badge/Pinal-red?label=Server" style="max-width: 100%;"></a>
<a href="https://huggingface.co/westlake-repl/Pinal"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yellow?label=Model" style="max-width: 100%;"></a>


The repository is an official implementation of Pinal: [Toward De Novo Protein Design from Natural Language](https://www.biorxiv.org/content/10.1101/2024.08.01.606258)

Quickly try our online server (16B) [here](http://www.denovo-pinal.com/)

If you have any questions about the paper or the code, feel free to raise an issue!

### Environment setup

Create and activate a new conda environment with Python 3.8.
```shell
conda create -n pinal python=3.8 --yes
conda activate pinal
pip install -r requirements.txt
```


### Download model weights

We provide a script to download the pre-trained model weights, as shown below. Please download all files and put them in the `weight` directory, e.g., `weights/Pinal/...`


```shell
huggingface-cli download westlake-repl/Pinal \
                         --repo-type model \
                         --local-dir weights/
```


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
> Warning: Inferencing with T2struc-15B requires at least 40GB GPU memory.


### Computational evaluation of the de novo designed proteins

For textual alignment, we recommend using [ProTrek](https://github.com/westlake-repl/ProTrek) to calculate the sequence-text similarity score.

For foldability, we recommend using pLDDT and PAE, outputted by [Alphafold series](https://golgi.sandbox.google.com/) or [ESMFold](https://github.com/facebookresearch/esm).


### Other resources

- [ProTrek](https://www.biorxiv.org/content/10.1101/2024.05.30.596740v2) and its [online server](http://search-protrek.com/)
- [Evola](https://www.biorxiv.org/content/10.1101/2025.01.05.630192v1) and its [online server](http://www.chat-protein.com/)
- [SaprotHub] (https://www.biorxiv.org/content/10.1101/2024.05.24.595648v5) and its [online server](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/SaprotHub_v2.ipynb?hl=en)

