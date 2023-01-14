python run_predict.py \
    --mode predict_dir \
    --pretrained_model weights/nisqa.tar \
    --data_dir /apdcephfs_cq3/share_2906397/users/thujunchen/enhanced_audio/dev_testset_sv56_norm/inference_wenzherefine_taijifreqgen_epoch15/devset \
    --num_workers 0 \
    --bs 10 \
    --output_dir /apdcephfs_cq3/share_2906397/users/thujunchen/enhanced_audio/test_nisqa