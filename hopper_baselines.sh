python baselines.py --env HopperVel-v1 --batch_size 16 --lr 1e-3 \
    --sigma 0.05 --max_steps 100 --max_epochs 200 --seq_len 5 \
    --output output/hopper --model_path saves/hopper --seed 123 \
    --robust --hist --so --maml






