python main.py --env Navigation2D-v2 --batch_size 16 --lr 5e-3 --sigma 0.05 \
    --max_steps 100 --max_epochs 500 --stage pretrain \
    --output output/navi_v2 --model_path saves/navi_v2 --seed 123   

python main.py --env Navigation2D-v2 --batch_size 16 --lr 5e-3 --sigma 0.05 \
    --max_steps 100 --max_epochs 500 --stage finetune \
    --output output/navi_v2 --model_path saves/navi_v2 --seed 123 \
    --FS --CA --IW_IES_N --IW_IES_Qu --IW_IES_Mix
    





