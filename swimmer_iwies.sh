python main.py --env SwimmerVel-v1 --batch_size 16 --lr 1e-2 --sigma 0.05 \
    --max_steps 100 --max_epochs 500 --stage pretrain \
    --output output/swimmer --model_path saves/swimmer --seed 123 

python main.py --env SwimmerVel-v1 --batch_size 16 --lr 1e-3 --sigma 0.05 \
    --max_steps 100 --max_epochs 200 --stage finetune \
    --output output/swimmer --model_path saves/swimmer --seed 123 \
    --FS --CA --IW_IES_N --IW_IES_Qu --IW_IES_Mix





