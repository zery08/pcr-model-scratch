# SPAS 계측 Item 예측 모델 개발 계획 (SSL 기반)

## 1) 목표
- **입력(X)**: `incoming_id`, `state_value`, `step_name`, `step_global_pos`, `param_mat`, `step_mask`
- **출력(Y)**: `wl_id`, `spas_item_id`, `wf_loc_id`, `wf_loc_x`, `wf_loc_y`, `y_value`, `y_mask`
- **핵심 목표**: Recipe 데이터를 활용해 `y_value`를 예측하고, 라벨이 적은 상황에서도 성능을 높이기 위해 SSL(Self-Supervised Learning) 사전학습 구조를 도입한다.

## 2) 데이터 단위 및 전처리
### 샘플 단위
- 한 샘플은 `incoming_id` 기준의 step 시퀀스 + 예측 조건(`spas_item_id`, `wl_id`, `wf_loc_*`)으로 구성한다.

### 전처리
- `step_name`: vocabulary 인덱싱(UNK 포함)
- `state_value`, `param_mat`: train split 기준 정규화/스케일링
- `step_global_pos`: positional feature로 변환
- `wf_loc_x`, `wf_loc_y`: wafer 좌표 정규화
- `step_mask`, `y_mask`: 모델 입력/손실 계산 시 마스킹에 사용

### 분할 전략
- 데이터 누수 방지를 위해 `incoming_id` 단위 split
- 권장 비율: Train / Valid / Test = 70 / 15 / 15

## 3) 모델 구조
### 3.1 Backbone (Recipe Encoder)
- Step-level 입력 임베딩
  - `step_name` embedding
  - `state_value` projection (MLP)
  - `param_mat` encoder (MLP)
  - `step_global_pos` positional encoding
- Sequence encoder
  - Transformer Encoder + `step_mask` 적용
- 출력: recipe representation `z_recipe`

### 3.2 Condition Encoder
- categorical embedding: `spas_item_id`, `wl_id`, `wf_loc_id`
- continuous projection: `wf_loc_x`, `wf_loc_y`
- 출력: condition representation `z_cond`

### 3.3 Regression Head
- `[z_recipe ; z_cond]` 결합 후 MLP 회귀
- 출력: `y_pred`
- 손실: `y_mask`를 반영한 masked regression loss

## 4) SSL 사전학습 전략
1. **Masked Step Modeling (MSM)**
   - step feature 일부를 mask 후 복원
2. **Contrastive Learning (InfoNCE)**
   - 동일 recipe 증강쌍은 positive, 다른 recipe는 negative
3. **(선택) Step Order Task**
   - 추후 확장 포인트로 유지

## 5) 학습 단계
1. **Stage A: SSL Pretrain**
   - Backbone 학습 (MSM + Contrastive)
2. **Stage B: Supervised Fine-tune**
   - Backbone + Condition Encoder + Regression Head end-to-end 학습
3. **Stage C: Ablation / Tuning**
   - `param_mat` 유무, 위치정보(`wf_loc_*`) 유무, loss 종류 비교

## 6) 손실함수/평가 지표
### 손실함수
- SSL: `L_ssl = λ1*L_msm + λ2*L_contrast (+ λ3*L_order)`
- Supervised: Masked Huber Loss(기본) 또는 Masked MSE

### 평가 지표
- 전체: MAE, RMSE, R²
- 조건별: `spas_item_id`, `wl_id`, wafer 위치 bin 별 성능
- 운영 관점: 고/중/저 `y_value` 구간별 오차 분석

## 7) 1차 구현 범위 현황
- [x] 데이터 로더/콜레이터 (mask 처리 포함) — `src/pcr_ssl/data.py`
- [x] Transformer 기반 backbone 구현 — `src/pcr_ssl/model.py`
- [x] SSL pretrain 루프 구현 — `scripts/train_ssl.py`
- [x] supervised fine-tune 루프 구현 — `scripts/train_supervised.py`
- [x] baseline(SSL 없음) 대비 성능 비교 실행 스크립트 — `scripts/run_pipeline.py`

## 8) 실행 방법
```bash
PYTHONPATH=src python scripts/run_pipeline.py
```

## 9) 기대효과
- 라벨 부족 환경에서 표현학습 성능 향상
- 새로운 `spas_item_id`/위치 조합에 대한 일반화 개선
- recipe step 변동성에 대한 robust한 예측
