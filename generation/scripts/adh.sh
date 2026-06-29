export PINAL_MODEL_ROOT=[your_pinal_model_root]
export T2struc_NAME=T2struc-15B

## Note that the T2struct temperature and argmax decoding of SaProt-T are relatively greedy for ADH. 
## Some outputs may share identical amino acid sequences, so after duplicate removal, the number of unique designs may be smaller than the target of 50,000.

accelerate launch generation/scripts/design.py\
    --desc_path generation/adh.txt\
    --temperature 0.8\
    --num 50000\
    --saprot_sample_type argmax\
    --output_dir generation/outputs

python generation/scripts/gather_distribution_results.py generation/outputs/adh

