# Powerlifting Causal Analysis

OpenPowerlifting 경기 기록을 이용해 체중이 선수의 생애 최고 Dots 점수에 미치는 영향을 성별로 분석한 프로젝트입니다. 원본 데이터 전처리부터 EMA 기반 선수별 지표 생성, 다중회귀, 매개효과, PSM/CEM, GPS-DR 분석까지 하나의 재현 가능한 흐름으로 구성되어 있습니다.

## 연구 질문

- 체중이 증가할수록 선수의 최고 Dots 점수는 어떻게 변하는가?
- 그 관계는 남성과 여성에게 동일한가?
- 초기 발전속도가 체중과 최고 Dots 점수 사이를 매개하는가?
- 회귀분석 결과가 매칭과 연속형 처치 인과추론에서도 유지되는가?

## 현재 실행 상태

- 분석 실행일: 2026-06-23
- 노트북: `main_code/powerlift_causal_analysis.ipynb`
- 실행된 코드 셀: 26개
- 실행 오류: 없음
- 실행 모드: `FULL_RUN = True`
- 무작위 시드: 42
- OLS·매개효과 부트스트랩: 1,000회
- PSM/CEM 임계값 민감도: 각 100회
- GPS-DR 부트스트랩: 각 100회
- 최종 분석 표본: 남성 3,960명, 여성 2,024명

실행 설정은 `outputs/run_metadata.json`에 기록되어 있습니다.

## 프로젝트 구조

```text
powerlift_causal_analysis/
├─ data/
│  ├─ openpowerlifting-2025-11-22-823f23d6.csv  # 원본 데이터
│  ├─ cleaned_sss.csv                            # 통합 전처리 데이터
│  ├─ cleaned_sss_M.csv                          # 남성 분석 입력
│  └─ cleaned_sss_F.csv                          # 여성 분석 입력
├─ preprocess_code/
│  └─ preprocess.py                              # 원본 → 분석 입력 전처리
├─ main_code/
│  └─ powerlift_causal_analysis.ipynb             # 메인 분석 노트북
├─ outputs/
│  ├─ derived/                                   # 선수 수준 EMA 파생 데이터
│  ├─ figures/                                   # 분석 그래프
│  ├─ tables/                                    # 반복 분석 상세 CSV
│  ├─ powerlifting_causal_analysis_results.xlsx  # 전체 결과 통합 문서
│  └─ run_metadata.json                          # 실행 설정과 표본 수
├─ pyproject.toml
├─ uv.lock
└─ README.md
```

`main.py`는 초기 프로젝트 템플릿이며 실제 분석 진입점이 아닙니다. 분석은 Jupyter 노트북에서 수행합니다.

## 환경 구성

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/) 사용을 권장합니다.

```powershell
cd C:\python\powerlift_causal_analysis
uv sync
```

주요 패키지는 다음과 같습니다.

- pandas, NumPy, SciPy
- statsmodels
- scikit-learn
- Matplotlib, Seaborn
- tqdm, openpyxl

정확한 버전 범위는 `pyproject.toml`, 잠금 버전은 `uv.lock`에서 확인할 수 있습니다.

## 1. 데이터 전처리

원본 데이터는 약 3.7백만 건, 42개 변수로 구성된 OpenPowerlifting 공개 데이터입니다.

프로젝트 루트에서 다음 명령을 실행합니다.

```powershell
uv run python preprocess_code\preprocess.py `
  --input data\openpowerlifting-2025-11-22-823f23d6.csv `
  --output-dir data `
  --overwrite
```

대용량 원본의 초기 단계는 기본 100,000행 단위 청크로 처리됩니다. 처리 중간 CSV를 보존하려면 `--keep-intermediate`를 추가합니다.

### 전처리 조건

1. `Equipment == Raw` 기록만 유지
2. Raw 기록 중 `Tested != Yes` 또는 결측 기록이 하나라도 있는 선수 전체 제외
3. `Sanctioned == Yes` 기록만 유지
4. 스쿼트·벤치프레스·데드리프트 기록이 모두 존재하고 0보다 큰 행만 유지
5. 선수별 기록이 6회 이상인 경우만 유지
6. 날짜를 이용해 나이 결측을 보간하고 성인 기록이 전혀 없는 선수 제외
7. `IPFCategory`, `Continent`, `Days_Since_Start` 생성
8. 동일 선수·동일 날짜 기록 병합: 숫자형은 평균, 문자형은 첫 값
9. 체중 결측과 `Place`가 `DD` 또는 `DQ`인 기록 제외
10. 6회 이상 조건을 다시 확인한 뒤 남성·여성 파일로 분리

### 전처리 출력

| 파일 | 행 수 | 용도 |
|---|---:|---|
| `data/cleaned_sss.csv` | 181,075 | 성별 통합 정제 데이터 |
| `data/cleaned_sss_M.csv` | 114,442 | 남성 메인 분석 입력 |
| `data/cleaned_sss_F.csv` | 66,631 | 여성 메인 분석 입력 |

결측값은 분석에 쓰이는 필수 변수 기준으로 제거됩니다. `Maturation_Slope`처럼 구간 정의가 불가능할 때 의도적으로 결측일 수 있는 선택 변수는 최종 모형에서 사용하지 않습니다. 각 회귀·매칭·GPS 단계에서도 해당 단계의 사용 변수에 대해 다시 결측을 제거합니다.

## 2. 메인 분석 실행

VS Code 또는 Jupyter 환경에서 아래 파일을 열고 프로젝트의 `.venv` Python 커널을 선택한 뒤 위에서 아래로 실행합니다.

```text
main_code/powerlift_causal_analysis.ipynb
```

노트북의 주요 설정:

```python
FULL_RUN = True
RANDOM_SEED = 42
REBUILD_ATHLETE_FEATURES = False
```

- `FULL_RUN=True`: 보고서용 반복 횟수로 전체 분석을 실행합니다.
- `FULL_RUN=False`: 구조 확인용으로 반복 횟수를 줄입니다.
- `REBUILD_ATHLETE_FEATURES=False`: `outputs/derived`의 선수 수준 캐시를 재사용합니다.
- 입력 CSV나 EMA 조건을 변경했다면 `REBUILD_ATHLETE_FEATURES=True`로 실행해야 합니다.

## 분석 방법

### 선수별 생애주기 지표

- 성별 체급 구간 적용
- 18세 이상, 체급별 6회 이상 출전
- 여러 체급에서 조건을 충족하면 가장 먼저 시작한 체급 한 개 선택
- 선수 기록을 30일 간격으로 선형보간
- 전체 경력 구간의 20%를 span으로 하는 EMA 적용
- 최고점이 마지막 시점인 선수는 아직 성장 중인 것으로 보고 제외
- 첫 연속 양의 기울기 구간에서 최대 기울기의 70% 이상인 구간으로 `Initial_Speed` 계산
- 경력 기간이 180일 미만인 선수 제외

그 결과 `outputs/derived/athlete_level_M.csv`와 `athlete_level_F.csv`가 생성됩니다.

### 통계 및 인과분석

- 기술통계, 분포, 상관관계, 체중–Peak Dots 산점도
- 다중회귀와 VIF
- 잔차–적합값 및 Q-Q plot 진단
- OLS 계수 1,000회 부트스트랩
- 초기 발전속도 간접효과 1,000회 부트스트랩
- PSM: 로지스틱 성향점수, 1:1 최근접 비복원 매칭
- CEM: 공변량 4구간 조대화 후 층화 고정효과
- 체중 Q1-Q3 범위에서 임계값을 무작위로 바꾸는 100회 민감도 분석
- GPS-DR: 랜덤포레스트 처치모형, KDE 기반 GPS, 안정화 가중치 1·99백분위 절단
- Restricted cubic spline과 100회 부트스트랩을 이용한 용량–반응 곡선

## 현재 분석 결과 요약

### 분석 표본

| 성별 | 선수 수 | 평균 체중 | 평균 Peak Dots | 평균 최고점 도달기간 |
|---|---:|---:|---:|---:|
| 남성 | 3,960 | 91.74 kg | 402.35 | 3.69년 |
| 여성 | 2,024 | 68.01 kg | 376.73 | 3.52년 |

### 다중회귀 핵심 계수

`E1`은 초기 발전속도를 제외한 Peak Dots 모형, `E2`는 초기 발전속도를 포함한 모형입니다.

| 성별 | 모형 | 변수 | 계수 | p-value | 부트스트랩 95% CI |
|---|---|---|---:|---:|---:|
| 남성 | E1 | 평균 체중 | +0.1764 | <0.001 | [0.0958, 0.2527] |
| 남성 | E2 | 평균 체중 | +0.1778 | <0.001 | [0.1048, 0.2560] |
| 남성 | E2 | 초기 발전속도 | +0.0292 | 0.426 | [-0.0374, 0.1039] |
| 여성 | E1 | 평균 체중 | -0.5343 | <0.001 | [-0.6901, -0.3775] |
| 여성 | E2 | 평균 체중 | -0.5272 | <0.001 | [-0.6754, -0.3722] |
| 여성 | E2 | 초기 발전속도 | +0.1332 | 0.0069 | [0.0327, 0.2352] |

통제변수를 고려한 회귀 결과에서 평균 체중 1kg 증가는 남성의 Peak Dots와 양의 관련, 여성의 Peak Dots와 음의 관련을 보였습니다.

### 매개효과

| 성별 | 평균 간접효과 | 95% CI | 유의성 |
|---|---:|---:|---|
| 남성 | -0.0011 | [-0.0049, 0.0021] | 유의하지 않음 |
| 여성 | -0.0069 | [-0.0189, 0.0015] | 유의하지 않음 |

두 성별 모두 간접효과 신뢰구간이 0을 포함하므로 초기 발전속도의 매개효과는 지지되지 않았습니다.

### PSM/CEM 임계값 민감도

| 성별 | 방법 | 평균 효과 | 95% CI | 유효 반복 |
|---|---|---:|---:|---:|
| 남성 | PSM | +5.377 | [2.297, 9.391] | 100 |
| 남성 | CEM | +1.782 | [-1.470, 5.389] | 100 |
| 여성 | PSM | -13.156 | [-21.970, -7.549] | 100 |
| 여성 | CEM | -18.034 | [-25.760, -14.211] | 100 |

이번 실행에서는 남성 PSM은 유의한 양의 효과를 보였지만 남성 CEM의 신뢰구간은 0을 포함했습니다. 여성은 두 방법 모두 음의 효과를 보였습니다. 따라서 남성 결과는 방법 선택에 더 민감하게 해석해야 합니다.

### GPS-DR 진단

| 성별 | 분석 N | 유효표본크기(ESS) | 최대 안정화 가중치 | 유효 부트스트랩 |
|---|---:|---:|---:|---:|
| 남성 | 3,960 | 3,434.62 | 2.50 | 100 |
| 여성 | 2,024 | 1,578.06 | 3.46 | 100 |

용량–반응 곡선과 GPS 가중 전후 균형은 각각 `gps_dr_dose_response.png`, `gps_balance_love_plots.png`에서 확인할 수 있습니다.

## 산출물 설명

### 통합 결과

`outputs/powerlifting_causal_analysis_results.xlsx`에는 다음 시트가 포함됩니다.

- 입력 데이터 개요와 품질 점검
- 선수 수준 기술통계와 상관관계
- 남녀 A·E1·E2 회귀계수와 VIF
- 매개효과와 PSM/CEM 요약
- GPS 진단, 공변량 균형, 용량–반응 곡선

### 상세 표

- `outputs/tables/M_PSM_threshold_iterations.csv`
- `outputs/tables/M_CEM_threshold_iterations.csv`
- `outputs/tables/F_PSM_threshold_iterations.csv`
- `outputs/tables/F_CEM_threshold_iterations.csv`
- `outputs/tables/M_GPS_DRF.csv`
- `outputs/tables/F_GPS_DRF.csv`

### 그래프

- `bodyweight_peakdots_scatter.png`: 체중과 Peak Dots 산점도
- `correlation_heatmaps.png`: 주요 변수 상관관계
- `ols_residual_diagnostics.png`: 회귀 잔차진단
- `matching_threshold_sensitivity.png`: PSM/CEM 임계값 민감도
- `psm_cem_love_plots.png`: 매칭 전후 공변량 균형
- `gps_dr_dose_response.png`: GPS-DR 용량–반응 곡선
- `gps_balance_love_plots.png`: GPS 가중 전후 균형
- `weightclass_summary.png`: 체급별 초기 발전속도와 Peak Dots
- `ema_trajectory_examples.png`: 선수별 EMA 궤적 예시

## 해석상의 주의점

- 이 결과는 관찰자료 기반이며 무작위 실험의 인과효과와 동일하지 않습니다.
- 훈련 방식, 영양, 신장, 골격근량, 체지방률 등 미측정 교란요인이 남아 있습니다.
- 체중은 시간에 따라 변하지만 분석은 선수의 선택된 체급 내 평균 체중을 처치로 사용합니다.
- PSM/CEM은 연속형 체중을 임계값으로 이분화하므로 정보 손실이 있습니다. 이를 보완하기 위해 GPS-DR을 함께 사용했습니다.
- 방법 간 결과가 다를 때는 한 방법만 선택하기보다 공변량 균형, 유효표본크기, 신뢰구간과 민감도 결과를 함께 판단해야 합니다.

## 데이터 및 저장소 관리

원본 CSV는 약 750MB이며 전처리·분석 산출물도 큽니다. 원격 저장소에 올릴 때는 데이터 라이선스와 저장 용량 제한을 먼저 확인하고, 필요하면 `data/`, `.venv/`, 대용량 `outputs/`를 Git 추적 대상에서 제외하십시오.

OpenPowerlifting 데이터 사용 조건과 최신 데이터 정보는 [OpenPowerlifting](https://www.openpowerlifting.org/)에서 확인할 수 있습니다.
