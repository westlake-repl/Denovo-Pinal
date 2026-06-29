export PINAL_MODEL_ROOT=[your_pinal_model_root]
export T2struc_NAME=T2struc-15B

accelerate launch generation/scripts/design.py\
    --desc_path generation/petase.txt\
    --temperature 1.2\
    --num 200000\
    --saprot_sample_type multinomial\
    --multinoimal_temperature 0.3 \
    --output_dir generation/outputs

python generation/scripts/gather_distribution_results.py generation/outputs/petase