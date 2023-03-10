data_dir="/apdcephfs/share_976139/users/thujunchen/data/dns5_blind/V5_BlindTestSet_mono/Track1_Headset/noisy"
preprocess_dir="/apdcephfs/share_976139/users/thujunchen/data/prepross_temp/BlindTestSet_Track1_Headset_noisy"
output_dir="/apdcephfs_cq3/share_2906397/users/thujunchen/exp_results/DNS5/nisqa/noisy"

python preprocess_wavfile.py \
    --input_dir ${data_dir} \
    --preprocess_dir ${preprocess_dir} \
    --sr 48000 \
    --chunk_lentgh 50

python run_predict.py \
    --mode predict_dir \
    --pretrained_model weights/nisqa.tar \
    --data_dir ${preprocess_dir} \
    --num_workers 0 \
    --bs 10 \
    --output_dir ${output_dir}

python eval_csv.py \
    --csv_pth ${output_dir}/NISQA_results.csv


rm -rf ${preprocess_dir}
