mkdir -p logs
ts=$(date '+%Y%m%d_%H%M%S')
log_file="logs/run.sh_$ts.log"
CUDA_VISIBLE_DEVICES=7
nohup python run_sparse_sim.py --gt_path /home/pub/lyb/history_qwen/crops_256/Cell_001_RawSIMData_gt_crop_90_132_256x256.png \
    --rounds 5 --blur_path /home/pub/lyb/history_qwen/crops_256/Cell_001_level_02_crop_90_132_256x256.png \
    --model_path /home/pub/bwm/models/Qwen2.5-VL-7B-Instruct \
    >> "$log_file" 2>&1 &
echo "Started: PID $!, logging to $log_file"