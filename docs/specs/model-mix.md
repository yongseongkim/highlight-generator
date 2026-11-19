# 모델 혼합 실행 SPEC

상태: **구현**, 2026-11-20. 공통 실행은 `highlight-discover`, 출력 겹쳐 보기는 `highlight-fusion`이다.
종목의 자동 경기 판단·구장 탐색은 계속 [종목 엔진 설계](README.md)의 후속 작업이다.

## 혼합 방법

| 관측 | 실행/입력 | 후보를 만드는 방법 |
| --- | --- | --- |
| OpenCV | 원본 전체 디코딩 | 대상 ROI 움직임과 카메라 변화 진단 |
| YOLO Pose | 선택적인 기존 포즈 캐시 | ROI 안 선수의 움직임 seed |
| YOLO 객체 | 선택적 `--object-model` | 포즈와 무관한 이미지 관측의 시간 변화 |
| SAM3 | 기존 독립 탐색·누락 확인·상세 예산 | 포즈와 무관한 이미지 관측의 시간 변화 |
| TrackNetV3 등 외부 모델 | 반복 가능한 `--model-evidence` | 같은 모델 실행·같은 원래 트랙의 직접 관측/시각 추적 변화 |

새 모델의 후보도 기존 후보와 합집합으로 보존하고, 출처별 후보를 번갈아 선택한다.
YOLO 객체·외부 모델의 움직임 후보도 남은 SAM 상세 예산에 제안하며 독립 scout 예약은 유지한다.
모델 신뢰도를 평균내거나, 여러 모델이 같은 객체를 봤다는 이유로 경기 결과를 확정하지 않는다.
검출은 종목 프로필의 원래 클래스만 후보에 반영하고, 다른 클래스도 원시 관측으로 저장한다.
공/셔틀의 2D 위치만으로 어느 코트 소속인지를 확정하지 않는다. 코트와 연결되지 않은 후보는 검토 대상으로 남긴다.

관측이 여러 번 있어도 보간·예측·정지 객체·장면 전환·긴 공백·다른 트랙 사이의 이동은 동작 seed로 사용하지 않는다.
선수는 공급된 ROI의 발 위치를 여유 폭 0으로 검사한다. 헤어핀 후보에 오버헤드 포즈를 요구하지 않는다.
기존 경기 단위의 시작·종료·이벤트·검증 상태는 보존하고, 길이 예산을 넘으면 전체를 건너뛴다.

## 모듈

| 파일 | 책임 |
| --- | --- |
| `discovery_objects.py` | YOLO 지연 로드, 독립 샘플 계획, CV와 디코딩 공유, 호출/시간 기록 |
| `model_evidence.py` | 외부 모델 JSON 검증, TrackNet CSV·상자 CSV 가져오기, 실행 출처 ID |
| `model_mix.py` | 디코더 시각 정렬, 모델별 관측 ID·카메라 구간·트랙 보존, seed와 커버리지 |
| `discovery_candidates.py` | 임의 모델 출처의 후보 합집합·예산 선택 |
| `fusion.py`, `fusion_signals.py` | 모델별 최근 관측을 겹쳐 표시하고 기존 ID로 검토 근거 출력 |

YOLO 객체 검출은 CV와 같은 프레임을 사용한다. 전체 호출 수는 `--max-object-calls`, 목표 주기는 `--object-fps`로 제한한다.
예산이 충분하면 인접한 두 관측을 영상 전반에 배치한다. 1~2회는 중앙 부근만 확인하므로 전체를 관측한 것으로 표시하지 않는다.
샘플 사이의 최대 공백을 출력한다. 연속 셔틀 궤적이 필요하면 전용 모델의 연속 입력을 별도로 확보해야 한다.
SAM3의 호출 상한·독립 scout 예약은 기존 계약을 유지한다. 객체 검출기는 SAM 단계 전에 해제한다.

## 기본 실행과 재사용

```sh
uv run highlight-discover input.mp4 --sport badminton --court court.json \
  --poses outputs/input_poses.npz --object-model models/yolo11n.pt \
  --object-fps 2 --max-object-calls 600 --sam-model models/sam3.pt \
  --max-sam-calls 90 --device mps --out-dir outputs/mixed

# SAM 캐시와 저장된 객체 관측 재사용. YOLO 객체 모델은 다시 로드하지 않습니다.
uv run highlight-discover input.mp4 --sport badminton --court court.json \
  --poses outputs/input_poses.npz --model-evidence outputs/mixed/model_evidence.json \
  --sam-cache outputs/mixed/sam-cache.jsonl --sam-model models/sam3.pt \
  --max-sam-calls 90 --out-dir outputs/mixed-replay

# 발견 보고서가 참조한 모든 모델 관측을 자동으로 읽습니다.
uv run highlight-fusion input.mp4 --sport badminton --court court.json \
  --poses outputs/input_poses.npz --candidates outputs/mixed/discovery.json \
  --sam-evidence outputs/mixed/sam_evidence.json --out-dir outputs/mixed-overlay
```

포즈 캐시가 없으면 `--poses`를 생략한다. 일반 COCO 가중치에는 셔틀 클래스가 없으므로 전용 셔틀 정확도를 주장하지 않는다.
`--plan-only`는 새 YOLO·SAM을 실행하지 않는다. 저장된 관측만으로도 후보 분석이 가능하다.
fusion은 별도 디코딩·출력 패스이며, 혼합 관측을 읽는 데 새 객체/SAM 추론은 필요하지 않다.

## TrackNetV3·SAMURAI 결과 가져오기

이 경로는 **이미 실행한 모델의 결과를 가져온다**. 새 모델의 설치·가중치 다운로드·학습·Mac 실행을 대신하지 않는다.

[TrackNetV3 원본 코드](https://github.com/qaz812345/TrackNetV3/blob/master/predict.py)의 CSV는
`Frame,Visibility,X,Y`를 사용하며 InpaintNet 후의 CSV도 같은 형식이다.
원본 추론 좌표와 보간 좌표를 CSV만으로 자동 구분할 수 없으므로 `--kind`를 명시해야 한다.
보간 전의 TrackNet 출력만 `observed`로 가져오며, 구분할 수 없는 후처리 출력은 전체를 `interpolated`로 취급한다.
위치 점은 원본 그대로 보존하고, 겹쳐 보기용 2px 상자는 실제 셔틀 크기로 사용하지 않는다.

```sh
uv run highlight-model-import input.mp4 tracknet-raw.csv --sport badminton \
  --format tracknet --model-name TrackNetV3 --checkpoint-id CHECKPOINT_HASH \
  --backend mps --label shuttlecock --kind observed --out outputs/tracknet.json

uv run highlight-discover input.mp4 --sport badminton --court court.json \
  --model-evidence outputs/tracknet.json --max-sam-calls 0 --out-dir outputs/tracknet-mix
```

`--backend`는 **외부에서 실제 사용한 실행 장치의 기록**이다. 이 예시가 TrackNetV3 MPS 실행 검증을 뜻하지 않는다.
입력은 원본 해상도·전역 좌표여야 한다. 시간으로 자른 영상은 `--frame-offset`에 원본 시작 프레임을 지정한다.
크기 변경·공간 crop·stride를 거친 출력은 먼저 원본 좌표와 프레임으로 명시적으로 변환해야 한다.

[SAMURAI 공식 demo](https://github.com/yangchris11/samurai/blob/master/scripts/demo.py)는 기본적으로 오버레이를 저장한다.
프레임별 마스크에서 계산한 `bbox_to_vis`를 다음 **교환용 CSV**로 내보내면 상자 입력으로 가져올 수 있다.
이 CSV가 SAMURAI의 기본 산출물이라는 뜻은 아니다. 여러 대상은 같은 프레임에 서로 다른 `track_id`로 기록한다.
마스크가 없으면 너비·높이 0으로 기록하며, 모델이 점수를 제공하지 않으면 `confidence` 열을 생략한다.

```csv
frame_index,x,y,width,height,track_id
0,30,55,10,40,A
1,35,55,10,40,A
```

```sh
uv run highlight-model-import input.mp4 samurai-boxes.csv --sport badminton \
  --format boxes --model-name SAMURAI --checkpoint-id SAM2_1_CHECKPOINT_HASH \
  --backend cuda --label person --kind tracked --out outputs/samurai.json
```

두 JSON을 `--model-evidence`로 각각 지정해 기존 YOLO·SAM3와 함께 사용한다.
점수가 없는 추적은 `confidence: null`을 유지하고 겹쳐 보기에 `unscored`로 표시한다.
후보의 관심도에만 낮은 기본 가중치를 쓰며 모델 신뢰도로 저장하지 않는다.

## 데이터·검토 계약

- 입력 JSON은 `schema_version`, `video`, `source_identity`, `sport`, 원본 `width/height`, `streams`를 가진다.
  각 stream은 `model`의 `name/checkpoint_id/backend/run_id`와 프레임별 `rows`를 제공한다.
  가져오기 도구는 이 구조를 생성한다. 외부의 체크포인트/장치 정보는 선언된 출처이며 실제 실행을 역검증하지 않는다.
- 관측 행은 `frame_index/time_s/time_basis/crop/objects`, 각 객체는 원래 `label/box/confidence/observation_kind/source_track_id`를 보존한다.
  같은 모델·실행에 서로 다른 데이터가 입력되면 새 `run_id`가 필요하다. 동일 파일을 반복 지정해도 중복 집계하지 않는다.
- `time_s`는 혼합 시 원본 전체 디코더의 해당 프레임 시각으로 맞추며 외부 시각은 `reported_time_s`로 남긴다.
  모델 간, 실행 간, 트랙 간 좌표를 연결하지 않는다. 카메라 구간이 달라지면 같은 외부 트랙 ID도 검토용 연결을 끊는다.
- `model_evidence.json`은 재사용 가능한 입력, `model_observations.json`은 현재 ROI와 연결한 원시 관측이다.
  `model_mix`는 출처별 seed 수와 점수 의미를 기록한다. 관심도와 경계 검증은 별개다.
- fusion은 발견 보고서의 모델 입력을 자동으로 읽고 원래 observation ID를 유지한다. 초록 상자는 최근 모델 관측이며
  0.12초 후 만료한다. 보간·예측은 원시 자료에 남기지만 관측 상자로 그리지 않는다.
- 모델 로드/새 추론/CV·디코딩/저장 관측 분석/출력을 따로 기록한다. 외부 실행 시간은 미확인이며 재사용 시간에 포함하지 않는다.

검증은 합성 영상·시험 검출기로 혼합 → 캐시 재실행 → 전체 오버레이 → 실제 근거 ID 검토를 검사했다.
실제 TrackNetV3·SAMURAI 추론, 종목별 정확도 향상과 Mac 처리량은 측정하지 않았다.
