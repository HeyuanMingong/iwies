### pretrain lr=1e-2, epochs=500; finetune lr=1e-3, epochs=200
python main.py --env HalfCheetahVel-v1 --batch_size 16 --lr 1e-2 --sigma 0.05 \
    --max_steps 100 --max_epochs 500 --stage pretrain \
    --output output/cheetah --model_path saves/cheetah --seed 123 \

python main.py --env HalfCheetahVel-v1 --batch_size 16 --lr 1e-3 --sigma 0.05 \
    --max_steps 100 --max_epochs 200 --stage finetune \
    --output output/cheetah --model_path saves/cheetah --seed 123 \
    --FS --CA --IW_IES_N --IW_IES_Qu --IW_IES_Mix





