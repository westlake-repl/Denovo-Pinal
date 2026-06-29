export PINAL_MODEL_ROOT=[your_pinal_model_root]
export T2struc_NAME=T2struc-15B

accelerate launch generation/scripts/design.py\
    --desc_path generation/h_protein.txt\
    --temperature 1.1\
    --num 50000\
    --saprot_sample_type multinomial\
    --multinoimal_temperature 0.1 \
    --output_dir generation/outputs

python generation/scripts/gather_distribution_results.py generation/outputs/h_protein