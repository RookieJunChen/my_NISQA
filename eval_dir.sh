data_dir="/apdcephfs_cq3/share_2906397/datasets/for_SSIC2023/blind_data"
output_dir="/apdcephfs_cq3/share_2906397/users/thujunchen/enhanced_audio/nisqa/blind_data"

python run_predict.py \
    --mode predict_dir \
    --pretrained_model weights/nisqa.tar \
    --data_dir ${data_dir} \
    --num_workers 0 \
    --bs 10 \
    --output_dir ${output_dir}

python eval_csv.py \
    --csv_pth ${output_dir}/NISQA_results.csv