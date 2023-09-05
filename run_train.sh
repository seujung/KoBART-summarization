python train.py --gradient_clip_val 1.0 \
                --max_epochs 2 \
                --checkpoint checkpoint \
                --accelerator gpu \
                --num_gpus 4 \
                --batch_size 32 \
                --num_workers 4