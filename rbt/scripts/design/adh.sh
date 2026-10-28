export PINAL_MODEL_ROOT=[your_pinal_model_root]
export T2struct_NAME=T2struct-15.5B

accelerate launch rbt/scripts/design.py\
    --desc_path rbt/desc/adh.txt\
    --temperature 1\
    --num 50000\
    --saprot_sample_type multinomial\
    --multinomial_temperature 0.3 \
    --output_dir rbt/outputs

python rbt/scripts/eval/gather_distribution_results.py rbt/outputs/adh