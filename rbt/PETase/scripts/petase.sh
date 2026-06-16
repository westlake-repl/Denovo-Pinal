cd /storage/yuanfajieLab/yuanfajie/fengyuan/Denovo-Pinal

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genesis

export PINAL_MODEL_ROOT=/storage/yuanfajieLab/yuanfajie/fengyuan/Denovo-Pinal/weights/Pinal
export T2struc_NAME=T2struc-15B

accelerate launch rbt/PETase/scripts/design.py\
    --desc_path rbt/PETase/petase.txt\
    --temperature 1\
    --num 50000\
    --saprot_sample_type multinomial\
    --multinoimal_temperature 0.3 \
    --output_dir /storage/yuanfajieLab/yuanfajie/fengyuan/Denovo-Pinal/rbt/PETase/outputs

python rbt/PETase/scripts/gather_distribution_results.py rbt/PETase/outputs/petase