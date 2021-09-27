# KoBART-summarization

## Data
- Data 구조
    - Train Data : 29,432
    - Valid Data : 7,358
    - Test Data : 9,182
 
## How to Train
- KoBART summarization fine-tuning
```
[Training]
python train.py --train True --test False --batch_size 16 --max_len 512 --lr 3e-05 --epochs 10

[Testing-rouge]
python train.py --train False --test True --batch_size 16 --max_len 512
```

## Model Performance
- Test Data rouge score
 
| | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|
| Precision| 0.5333 | 0.3463|0.4534|
| Recall| 0.5775| 0.3737|0.4869|
| F1| 0.5381| 0.3452|0.4555|

## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)
