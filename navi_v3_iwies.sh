### pretrain lr=1e-2, epochs=500; finetune lr=1e-3, epochs=1000
python main.py --env Navigation2D-v3 --batch_size 16 --lr 1e-3 --sigma 0.05 \
    --max_steps 100 --max_epochs 500 --stage pretrain \
    --output output/navi_v3 --model_path saves/navi_v3 --seed 123 

python main.py --env Navigation2D-v3 --batch_size 16 --lr 1e-3 --sigma 0.05 \
    --max_steps 100 --max_epochs 500 --stage finetune \
    --output output/navi_v3 --model_path saves/navi_v3 --seed 123 \
    --CA --IW_IES_N --IW_IES_Qu --IW_IES_Mix --FS





