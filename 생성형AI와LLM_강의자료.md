# 생성형 AI와 LLM

**공간정보빅데이터학과**

---

## 1. 생성형 AI란 무엇인가

기존의 판별 모델(Discriminative Model)이 P(Y|X), 즉 입력 X가 주어졌을 때 레이블 Y를 예측하는 데 집중했다면, 생성형 모델(Generative Model)은 데이터 자체의 분포 P(X)를 학습하여 새로운 데이터를 "생성"할 수 있습니다.

원격탐사 수업에서 배운 토지피복 분류(Land Cover Classification)는 전형적인 판별 모델입니다 — 픽셀이 주어지면 "산림"인지 "도심"인지 분류합니다. 반면 생성형 모델은 "산림처럼 보이는 위성영상 패치를 새로 만들어내라"는 과제를 수행합니다.

### 생성형 AI의 계보

- **GAN** (Generative Adversarial Network, 2014) — 생성자와 판별자의 적대적 학습
- **VAE** (Variational Autoencoder) — 잠재공간(latent space) 기반 확률적 생성
- **Diffusion Model** — 노이즈 제거 과정을 통한 생성 (현재 이미지 생성의 주류)
- **Transformer 기반 LLM** — 텍스트/시퀀스 생성의 주류

---

## 2. LLM의 핵심 원리: Transformer와 Self-Attention

### 2.1 왜 Transformer인가

2017년 "Attention is All You Need" 논문 이전에는 RNN/LSTM이 순차 데이터 처리의 표준이었습니다. 문제는 순차 처리로 인한 병렬화 불가능성과 장거리 의존성(long-range dependency) 소실이었습니다.

Transformer는 **Self-Attention 메커니즘**으로 이를 해결합니다. 수식으로 보면:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 Query, Key, Value는 각 토큰 임베딩에 학습된 가중치 행렬을 곱해 얻습니다. 이 구조 덕분에 문장 내 모든 토큰이 서로 직접 관계를 계산할 수 있고, 병렬 연산이 가능해집니다.

> **공간정보 전공자를 위한 비유**: 이는 마치 공간 자기상관(Spatial Autocorrelation)을 계산할 때 모든 지점 쌍(pairwise) 간의 가중치를 동시에 고려하는 것과 유사합니다. Moran's I가 공간 가중행렬 W를 통해 지점 간 관계를 정량화하듯, Self-Attention은 학습된 가중치로 토큰 간 관계를 정량화합니다.

### 2.2 LLM의 학습 과정

1. **사전학습(Pre-training)**: 대규모 텍스트 코퍼스에서 다음 토큰 예측(Next Token Prediction)이라는 자기지도학습(Self-Supervised Learning) 수행
2. **지도 미세조정(SFT, Supervised Fine-Tuning)**: 사람이 작성한 질문-답변 쌍으로 조정
3. **인간 피드백 강화학습(RLHF)**: 사람의 선호도를 반영해 보상모델을 학습하고 이를 기준으로 정책을 최적화

### 2.3 스케일링 법칙(Scaling Laws)

모델 파라미터 수, 데이터셋 크기, 연산량(compute) 세 가지가 로그-선형 관계로 성능(손실함수)과 연결된다는 것이 경험적으로 밝혀졌습니다(Kaplan et al., Chinchilla). 이것이 "더 큰 모델이 더 똑똑하다"는 최근 산업 트렌드의 이론적 근거입니다.

---

## 3. 공간정보 분야와의 접점 (Geospatial × LLM/GenAI)

### 3.1 GeoAI와 멀티모달 LLM

최근 위성/항공영상 인식에 Vision-Language Model(VLM)이 결합되면서, "자연어로 위성영상 질의하기"(예: "이 영상에서 침수 지역을 찾아줘")가 가능해지고 있습니다. CLIP, GPT-4V류 모델의 지리공간 특화 버전(예: RS-CLIP, GeoChat)이 활발히 연구되고 있습니다.

### 3.2 공간 추론(Spatial Reasoning)의 한계

LLM은 텍스트 기반 통계 패턴 학습이기 때문에, 실제 좌표계, 거리, 위상관계(topology) 추론에는 근본적 취약점이 있습니다. "서울에서 부산까지 몇 km인가"는 학습 데이터에 있으면 맞히지만, 임의의 두 좌표 간 최단 경로 네트워크 분석 같은 정밀한 공간연산은 GIS 엔진(PostGIS, ArcPy 등)과의 **도구 연동(Tool Use / Function Calling)** 없이는 신뢰하기 어렵습니다.

### 3.3 자연어 기반 GIS 인터페이스

LLM을 자연어-SQL(Text-to-SQL) 변환기로 활용해 PostGIS 공간질의를 생성하거나, 자연어 명령으로 지도 시각화를 자동 생성하는 연구가 활발합니다.

### 3.4 합성 데이터 생성

Diffusion 모델을 활용해 희귀 재해 상황(산불, 홍수)의 합성 위성영상을 생성, 데이터 증강(Data Augmentation)에 활용하는 연구도 진행 중입니다.

---

## 4. 한계와 비판적 시각

- **환각(Hallucination)**: 통계적 패턴 생성이므로 사실이 아닌 내용을 자신 있게 생성할 수 있습니다. 공간데이터 관련 질의에서는 좌표, 면적, 지명 등 정량적 사실 오류에 특히 주의해야 합니다.
- **최신성 문제**: 학습 데이터의 시점 이후 정보는 알 수 없어, 실시간 공간정보(교통, 재해)에는 RAG(Retrieval-Augmented Generation)나 실시간 API 연동이 필수적입니다.
- **연산·환경 비용**: 대규모 모델 학습의 탄소발자국 문제도 학계에서 논의되고 있습니다.
