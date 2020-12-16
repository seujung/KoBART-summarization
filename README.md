# KoBART-summarization

## Install KoBART
```
git submodule update --init --recursive
cd KoBART
pip install -r requirements.txt
pip install .
```

## Requirements
```
pytorch==1.7.0
transformers==4.0.0
pytorch-lightning==1.1.0
streamlit==0.72.0
```
## Data
- [Dacon 한국어 문서 생성요약 AI 경진대회](https://dacon.io/competitions/official/235673/overview/) 의 학습 데이터를 활용함
- 학습 데이터에서 임의로 Train / Test 데이터를 생성함
- 데이터 탐색에 용이하게 tsv 형태로 데이터를 변환함
- Data 구조
    - Train Data : 34,242
    - Test Data : 8,501
  
| news  | summary |
|-------|--------:|
| 뉴스원문| 요약문 |  


## How to Train
- KoBART summarization fine-tuning
```
./prepare.sh
pip install -r requirements.txt
python train.py  --gradient_clip_val 1.0 --max_epochs 50 --default_root_dir logs  --gpus 1 --batch_size 4
```
## Generation Sample
| ||Text|
|-------|:--------|:--------|
|1|Label|태왕의 '성당 태왕아너스 메트로'모델하우스는 초역세권 입지와 변화하는 라이프스타일에 맞춘 혁신평면으로 오픈 당일부터 관람객의 줄이 이어지면서 관람객의 호평을 받았다.|
|1|koBART|아파트 분양시장이 실수요자 중심으로 바뀌면서 초역세권 입지와 변화하는 라이프스타일에 맞춘 혁신평면이 아파트 선택에 미치는 영향력이 커지고 있는 가운데, 태왕이 지난 22일 공개한 ‘성당 태왕아너스 메트로’ 모델하우스를 찾은 방문객들은 합리적인 분양가와 중도금무이자 등의 분양조건도 실수요자에게 유리해 높은 청약경쟁률을 기대했다.|

| ||Text|
|-------|:--------|:--------|
|2|Label|광주지방국세청은 '상생하고 포용하는 세정구현을 위한' 혁신성장 기업 세정지원 설명회를 열어 여러 세정지원 제도를 안내하고 기업 현장의 애로, 건의사항을 경청하며 기업 맞춤형 세정서비스를 제공할 것을 약속했다.|
|2|koBART|17일 광주지방국세청은 정부광주지방합동청사 3층 세미나실에서 혁신성장 경제정책을 세정차원에서 뒷받침하기 위해 다양한 세정지원 제도를 안내하는 동시에 기업 현장의 애로·건의사항을 경청하기 위해 ‘상생하고 포용하는 세정구현을 위한’ 혁신성장 기업 세정지원 설명회를 열어 주목을 끌었다.'|

| ||Text|
|-------|:--------|:--------|
|3|Label|신용보증기금 등 3개 기관은 31일 서울 중구 기업은행 본점에서 최근 경영에 어려움을 겪는 소상공인 등의 금융비용 부담을 줄이고 서민경제에 활력을 주기 위해 '소상공인. 자영업자 특별 금융지원 업무협약'을 체결했다고 전했으며 지원대상은 필요한 조건을 갖춘 수출중소기업, 유망창업기업 등이다.|
|3|koBART|최근 경영애로를 겪고 있는 소상공인과 자영업자의 금융비용 부담을 완화하고 서민경제의 활력을 제고하기 위해 신용보증기금·기술보증기금·신용보증재단 중앙회·기업은행은 31일 서울 중구 기업은행 본점에서 ‘소상공인·자영업자 특별 금융지원 업무협약’을 체결했다.|



## Model Performance
- Test Data 기준으로 rouge score를 산출함
- Score 산출 방법은 Dacon 한국어 문서 생셩요약 AI 경진대회 metric을 활용함
  
| | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|
| Precosion| 0.515 | 0.351|0.415|
| Recall| 0.538| 0.359|0.440|
| F1| 0.505| 0.340|0.415|

## Demo
- 학습한 model binary 추출 작업이 필요함
   - pytorch-lightning binary --> higgingface binary로 추출 작업 필요
   - hparams의 경우에는 <b>./logs/tb_logs/default/version_0/hparams.yaml</b> 파일을 활용
   - model_binary 의 경우에는 <b>./logs/kobart_summary-model_chp</b> 안에 있는 .ckpt 파일을 활용
   - 변환 코드를 실행하면 <b>./summary_binary</b> 에 model binary 가 추출 됨
  
```
 python get_model_binary.py --hparams hparam_path -- model_binary model_binary_path
```

- streamlit을 활용하여 Demo 실행
    - 실행 시 <b>http://localhost:8501/</b> 로 Demo page가 실행됨
```
streamlit run infer.py
```

- Demo Page 실행 결과
  - [원문링크](https://www.mk.co.kr/news/society/view/2020/12/1289300/?utm_source=naver&utm_medium=newsstand)
  
<img src="imgs/demo.png" alt="drawing" style="width:600px;"/>

## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-chatbot](https://github.com/haven-jeon/KoBART-chatbot)