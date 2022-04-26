### pretrain in the original environment
python main.py --env Navigation2D-v1 --batch_size 16 --lr 1e-2 --sigma 0.05 \
    --max_steps 100 --max_epochs 200 --stage pretrain \
    --output output/navi_v1 --model_path saves/navi_v1 --seed 123 \
    --num_workers 16

### incremental learning in the new environment
python main.py --env Navigation2D-v1 --batch_size 16 --lr 1e-3 --sigma 0.05 \
    --max_steps 100 --max_epochs 200 --stage finetune --seed 123 \
    --output output/navi_v1 --model_path saves/navi_v1 --num_workers 16 \
    --CA --IW_IES_Qu --IW_IES_N --IW_IES_Mix --FS






