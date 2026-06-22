export PINAL_MODEL_ROOT=[your_pinal_model_root]
export T2struc_NAME=T2struc-15B

accelerate launch rbt/scripts/design.py\
    --desc_path rbt/adh.txt\
    --temperature 1\
    --num 50000\
    --saprot_sample_type multinomial\
    --multinoimal_temperature 0.3 \
    --output_dir rbt/outputs

python rbt/scripts/gather_distribution_results.py rbt/outputs/adh