python baselines.py --env Navigation2D-v1 --batch_size 16 --lr 1e-3 \
    --sigma 0.05 --max_steps 100 --max_epochs 200 --seq_len 5 \
    --output output/navi_v1 --model_path saves/navi_v1 --seed 123 --device cpu \
    --robust --hist --so --maml





