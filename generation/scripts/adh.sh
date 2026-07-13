export PINAL_MODEL_ROOT=[your_pinal_model_root]
export T2struc_NAME=T2struc-15B


accelerate launch generation/scripts/design.py\
    --desc_path generation/adh.txt\
    --temperature 0.8\
    --num 2500\
    --saprot_sample_type argmax\
    --output_dir generation/outputs

python generation/scripts/gather_distribution_results.py generation/outputs/adh

