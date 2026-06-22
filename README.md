# ✈️ 항공기 운항 시스템 기반 정시성 개선 및 운항 지연 요인 분석 연구
> **2026 고용노동부 미래내일 일경험 프로젝트형 (이스타항공 참여)**
> Flightradar24 및 항공 운항 데이터를 융합하고 PyTorch 기반 Deep Learning(MLP) 알고리즘을 적용한 데이터 기반 예지 정비(Predictive Maintenance) 및 정시성 최적화 시스템입니다.

---

## 📅 Project Timeline
- **진행 기간**: 2026년 3월 ~ 2026년 6월

---

## 👥 Team Members & Roles
| 이름 | 소속 | 역할 | 담당 업무 |
| :---: | :---: | :---: | :--- |
| **황정현** | 인하대학교 | **팀장** | 프로젝트 총괄, 일정 관리 및 최종 결과 검토 |
| **김소희** | 인하대학교 | **팀원 (ML/DL)** | **딥러닝 예측 모델 선정·설계, 모델 학습 및 추론 결과 해석** |
| **김승찬** | 인하대학교 | 팀원 | 도메인/정시성 자료 조사, 예방 정비 운영 개선안 도출 |
| **김승현** | 인하대학교 | 팀원 | 운항 이종 데이터 수집·정제 및 전처리 데이터셋 가공 |

---

## 🚀 Key Features & Technologies

### 1. Data Engineering & Ingestion
- **이종 데이터 융합 매핑 (Data Fusion)**
  - 이스타항공 내부 운항 데이터와 글로벌 실시간 항공기 추적 데이터(Flightradar24)를 기재별 시계열로 추적 및 매핑하는 독창적 전처리 프로세스 정립
- **극단적 클래스 불균형(Data Imbalance) 해소**
  - 전체 약 5,500건 중 정비지연(1) 케이스가 40여 건(1% 미만)에 불과한 극단적 불균형을 극복하기 위해 **SMOTE (Synthetic Minority Over-sampling Technique)** 알고리즘 적용
  - 기계적 단순 복사를 탈피하고 K-최근접 이웃(K-NN) 간 직선 세그먼트 상에 가상 합성 데이터를 생성하여 Train Set의 클래스 비율을 1:1로 최적화
- **도메인 기반 4대 핵심 변수(Feature) 추출**
  - 기재 노후도 및 역학적 피로도를 대변하는 **[이착륙 횟수, 총 비행시간]**
  - 계절 및 기후 변수를 반영하는 **[운항 월(Month)]**
  - 고유 노선 및 운항 패턴의 규칙성을 추적하는 **[편명]**

### 2. Deep Learning Architecture (PyTorch)
- **Multi-Layer Perceptron (MLP) 기반 이진 분류 네트워크**
  - 4차원 입력 벡터($x \in \mathbb{R}^4$)를 수용하는 입력층 설계
  - 은닉층 1: 4차원 $\rightarrow$ 32차원 확장 (Linear Layer) + 비선형 활성화 함수 **ReLU** 적용하여 표현력 확보
  - 은닉층 2: 32차원 $\rightarrow$ 16차원 다운샘플링 연산을 통한 가중치 병목 제어
  - 출력층: 1차원 선형 레이어 + **Sigmoid 활성화 함수** 배치를 통해 0.0 ~ 1.0 사이의 정량적 정비지연 발생 확률(%) 도출
- **모델 과적합(Overfitting) 제어 규제화 기법**
  - 은닉층 간 **Dropout 레이어 교차 배치** (Layer 1 뒤 50%, Layer 2 뒤 30%)로 뉴런 간 상호 의존성 제거
  - **Adam Optimizer** 내 가중치 감쇄(Weight Decay = $1\times10^{-4}$) 파라미터를 선언하여 L2 Norm 크기를 제한, 검증 손실(Validation Loss) 하향 안정화
- **동적 브레이크 메커니즘 (Early Stopping)**
  - 매 에폭마다 `BCELoss (Binary Cross Entropy Loss)` 추이를 실시간 모니터링
  - 최적 가중치 텐서를 `best_model.pth`로 로컬에 자동 갱신하되, 검증 손실이 20에폭(`Patience=20`) 동안 정체/개악될 경우 과적합 임계점으로 판단하여 학습 자동 종료 (실제 실험 시 81 에폭에서 수렴 완료)

---

## ⚙️ System Operation & Logic

프로젝트 초기 구상안이었던 '운항 시간 조정 모델'이 공항 슬롯 및 타 항공사 스케줄 등 외부 외생 변수 제약으로 실무 적용이 불가함을 본사 피드백을 통해 인지하고, 항공사 통제 가능한 내부 인자 중심의 **'예지 정비(Predictive Maintenance) 모델'**로 패러다임 전환을 달성했습니다.

```text
[운항 정보 입력] (이착륙 횟수, 총 비행시간, 운항예정 월, 편명 한글/영문 자동 파싱)
       │
       ▼
[실시간 정규화] (학습에 사용된 StandardScaler 객체를 통해 실시간 Transform 처리)
       │
       ▼
[신경망 연산] (PyTorch FloatTensor 변환 후 MLP 및 Sigmoid 순방향 연산)
       │
       ▼
[임계치 필터링] (안전 마진 확보를 위한 임계치 Threshold = 0.5 적용)
       │
 ┌─────┴─────┐
 ▼ (50% 미만)  ▼ (50% 이상)
[정상 비행 가능] [정비지연 위험 수치 발생] -> 사전 예방 정비 강력 권고 메시지 출력


├── src/
│   ├── model/
│   │   └── mlp_network.py     # PyTorch 기반 다층 퍼셉트론 모델 정의
│   ├── preprocess.py          # SMOTE 오버샘플링 및 StandardScaler 전처리 스크립트
│   └── inference.py           # 현장 관리자용 시나리오 입력 및 위험도 산출 메인 루틴
├── docs/
│   ├── 미래내일_일경험_프로젝트형_결과보고서_이스타항공.pdf
│   └── best_model.pth         # 최적 학습 가중치 텐서 파일
└── README.md
