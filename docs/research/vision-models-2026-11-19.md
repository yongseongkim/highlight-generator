# 스포츠 하이라이트용 비전 모델 조사

공식 저장소·문서 확인: **2026-11-19**. 아래의 도입 우선순위는 이 프로젝트의 영상과 Mac 환경에 대한
설계 판단이며, 새 모델의 정확도·속도를 실측한 결과가 아니다. 새 가중치를 내려받거나 추론하지 않았다.
종목별 사용 조건은 [배드민턴](../specs/badminton.md), [농구](../specs/basketball.md),
[축구](../specs/football.md), 공통 데이터 계약은 [종목 SPEC](../specs/README.md)을 따른다.

## 추천 구성

**대상 발견 → 위치·추적 → 구장 좌표 → 종목 이벤트 → 관심도 판단**을 서로 다른 단계로 둔다.
SAM3·SAMURAI만으로 경기 규칙, 타구, 득점, 완결된 랠리를 모두 알아내지는 못한다.

| 역할 | 먼저 비교할 도구 | 이 프로젝트에서의 판단 |
| --- | --- | --- |
| 포즈 밖의 대상 발견 | 기존 SAM3 이미지, YOLOE | 독립 시간 예산을 유지한다. 새로운 표적을 찾는 경로 |
| 지정한 대상의 연속 추적 | 기존 ByteTrack, 필요시 SAM2.1/SAMURAI·BoT-SORT | 단순한 상자 연결부터 시작하고 가림·마스크가 필요한 구간만 확장 |
| 배드민턴 셔틀 | TrackNetV3 | 작은 셔틀에 특화된 우선 실험. 연속 입력과 보간의 구분이 필요 |
| 농구 코트·림·공 | OpenCV 규격 검증 + KaliCalib/DeepSportRadar 보정 비교, 전용 공·림 검출 | 평면 코트와 높이 있는 림을 별도 취급 |
| 축구 구장·행동 | PnLCalib/TVCalib, 공·선수 검출, T-DEED 계열 | 팬·줌 보정과 공격/득점 후보의 시간 검출을 분리 |
| 점수 상황·후보 검토 | 점수판 OCR, 제한된 Qwen3-VL 검토 | 마지막 보조 단계. 기존 근거 ID와 불확실성을 유지 |

한 번의 전체 실행에서 모든 모델을 켜는 구성은 우선안이 아니다. 같은 입력·예산에서 각 단계의 개선 효과를
따로 비교한 후 채택한다. 특히 현재는 **잘못된 대상 코트 보정을 먼저 고치는 작업**이 필요하다.

## SAM3, SAM2.1, SAMURAI의 차이

| 모델 | 입력·역할 | 적용 후보 | 제약과 Mac 상태 |
| --- | --- | --- | --- |
| [SAM3](https://github.com/facebookresearch/sam3) | 텍스트/예시로 개념에 맞는 객체를 발견·분할하고 비디오에서 추적 | 포즈가 놓친 선수·장비의 독립 탐색, 표면 마스크 보조 | 현재 프로젝트에서 실행한 것은 Ultralytics 이미지 경로의 **상자 관측**이다. 마스크·공식 비디오 경로는 미연동 |
| [SAM 3.1](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) | 여러 객체를 함께 처리하는 비디오 추적 개선 | 추후 긴 다중 객체 추적 비용 비교 | 공식 H100 다중 객체 수치는 Mac 실측이 아니다. 현재 이미지 호출만 교체한다고 같은 개선이 생기지 않음 |
| [SAM2.1](https://github.com/facebookresearch/sam2) | 점·상자 등으로 지정한 대상을 메모리와 함께 영상에서 분할·추적 | 특정 선수/장비의 짧은 마스크 추적 | 시작 표적 지정과 재발견 경로 필요. 이 저장소의 MPS 비디오 실행·성능은 미검증 |
| [SAMURAI](https://github.com/yangchris11/samurai) | SAM2.1에 운동 예측·메모리 선택을 결합한 zero-shot 추적 | 가려지는 지정 대상의 추적을 SAM2.1과 비교 | 공식 사용자 영상 예제는 첫 프레임 상자 필요. 텍스트로 전 화면의 새 객체를 발견하는 경로를 대신하지 않음. MPS 미검증 |

SAMURAI는 SAM2.1 가중치를 활용하며 추가 학습 없이 추적하는 접근이다.
급격한 타구 방향 전환·작은 크기·모션 블러가 많은 셔틀에서 더 좋다는 결론은 아직 없다.
셔틀은 TrackNetV3와 비교하고, SAMURAI는 선수·더 큰 표적의 가림 복구부터 평가하는 편이 적절하다.
추적기가 잃은 표적은 독립 발견 경로로 다시 찾고, 재등장 ID가 이전 실제 객체와 같은지는 별도로 판단한다.

[SAM3 공식 설치](https://github.com/facebookresearch/sam3#installation)는 Python 3.12 이상·PyTorch 2.7 이상·CUDA 12.6 이상을
요구한다. 이 조건은 현재 Ultralytics를 통한 Mac 이미지 실행과 다르다. 공식 저장소를 그대로 설치하는 작업을
기존 MPS 경로의 단순 업그레이드로 취급하지 않는다. SAM3 가중치의 접근 승인도 필요하다.

## 검출·포즈·작은 공·일반 추적

| 후보와 공식 자료 | 입력·출력 / 적합한 역할 | 도입 전에 확인할 점 |
| --- | --- | --- |
| 기존 YOLO11 Pose·객체 검출 | 선수 관절, 학습된 클래스의 상자. 현재 MPS 기준선 | COCO의 `sports ball`, `tennis racket`이 셔틀·배드민턴 라켓·림을 모두 지원한다는 뜻은 아니다. 체크포인트 원래 라벨 유지 |
| [YOLOE](https://docs.ultralytics.com/models/yoloe/) | 텍스트/시각 프롬프트를 쓰는 개방 어휘 검출·분할 | SAM3 독립 발견 비용과 비교할 후보. 작은 셔틀·원거리 공에 대한 도메인 정확도와 선택한 버전의 MPS 실행은 미검증 |
| [TrackNetV3](https://github.com/qaz812345/TrackNetV3) | 연속 영상에서 셔틀 위치, 별도 InpaintNet으로 누락 경로 보완 | 공식 가중치 링크 제공. 체크포인트의 입력 길이·배경 모드·학습 시각 간격을 지켜야 한다. 기존 CUDA 중심 코드의 MPS 이식 필요 여부 확인 |
| [RTMPose / MMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) | 사람 crop에서 관절. 기존 YOLO Pose의 대안 비교 | 보통 사람 검출 단계도 필요하므로 포즈 누락 문제가 저절로 사라지지 않는다. [MMDeploy 경로](https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/how_to_deploy.md)에 Core ML 등이 있어도 실제 선택 모델의 변환·정확도·Mac 비용은 별도 검증 |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | 검출 상자의 시간 연결, 낮은 점수 검출도 연결에 활용 | 현재 CPU 연결 기준선. 검출이 전혀 없는 대상을 새로 발견하거나 선수 실명을 식별하지 않음 |
| [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | 상자 연결 + 카메라 움직임 보정, 선택적 외형 ReID | 팬·줌 영상과 선수 가림의 비교 후보. 카메라 보정과 ReID 설정·비용을 분리하며 MPS ReID 경로는 미검증 |

TrackNetV3의 궤적 보완은 유용하지만, 보완된 좌표로 타구·바닥 접촉을 검증하면 안 된다.
YOLO/SAM의 검출 상자, TrackNet의 직접 관측, InpaintNet/Kalman의 보간·예측을 구분한다.
모든 추적은 source frame/PTS와 관측 공백을 유지하고, 모델 내부 프레임 인덱스를 원본 시각으로 오인하지 않는다.

## 구장 보정·종목 행동 모델

| 후보와 공식 자료 | 맡길 수 있는 일 | 이 영상들에 적용할 때의 제한 |
| --- | --- | --- |
| [KaliCalib: DeepSportRadar 2022 우승 코드](https://github.com/DeepSportradar/2022-winners-camera-calibration-challenge) | 농구장 전용 카메라/코트 보정 연구 | 범용 SAM보다 과제에 직접 맞는 비교 후보. 가중치 배포·전처리·카메라 가정·현재 실행 환경을 확인해야 하며 현재 미연동 |
| [DeepSportRadar 카메라 보정 baseline](https://github.com/DeepSportradar/camera-calibration-challenge) | FIBA 코트의 선 클래스 분할 → 교차점 → 카메라 보정. 사전 학습 가중치 링크 제공 | 왜곡 보정된 입력 가정. 교차점이 부족하면 학습 카메라 평균값을 반환하는 원본 fallback은 이 프로젝트에서 **성공 보정으로 수용하지 않음** |
| [DeepSportRadar ball-3D 과제](https://github.com/DeepSportradar/ball-3d-localization-challenge) | 주어진 공 중심과 공 크기·카메라 정보를 이용한 3D 위치 추정 baseline | 입력에 정답 공 중심을 주는 과제다. 원본 영상에서 공을 자동 발견하는 완성된 검출기로 간주하지 않음 |
| [PnLCalib](https://github.com/mguti97/PnLCalib) | 축구의 점·선 검출과 카메라 추정/정제. 공개 사전 학습 가중치 제공 | 축구장 보정 우선 후보. 부분 시야·줌·렌즈 왜곡과 동호회 고정 카메라에서 검증. 다른 종목의 코트로 무학습 전용하지 않음 |
| [TVCalib](https://github.com/MM4SPA/tvcalib) | 축구 선 분할과 투영 최적화. 공개 분할 가중치 제공 | PnLCalib와 정확도·시간·라이선스 조건을 비교. 낮은 투영 오차만으로 올바른 구장 ID가 보장되지는 않음 |
| [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate) | 선수 추적·팀/역할·필드 좌표를 연결한 참고 파이프라인 | 하나의 모델이 아니다. PnLCalib 등을 조합한 모듈 구조 참고; 전체 의존성과 비용을 Mac 기본값으로 가져오지 않음 |
| [T-DEED](https://github.com/arturxe2/T-DEED), [SoccerNet Team Spotting](https://github.com/SoccerNet/sn-teamspotting) | 영상의 시간적 이벤트 spotting, 후자는 행동 팀까지 예측하는 확장. 공개 가중치 링크 제공 | 축구 이벤트 후보 생성에 유용할 가능성. 지원 이벤트 라벨·학습 도메인·연속 입력 구성을 확인. 모델이 `goal`을 예측해도 공식 결과와 검증된 경계가 되는 것은 아님 |

농구장의 원은 투영하면 일반적으로 원뿔곡선이 된다. 축구장은 팬·줌이 있고, 배드민턴장은 여러 코트와 다른 종목의
선이 겹칠 수 있다. 따라서 동일한 사각형 검출기를 세 종목에 그대로 쓰지 않는다.
선의 의미·간격·교차 구조를 종목 템플릿에 맞추고, 보정에 쓰지 않은 선으로 검증한다.
SAM으로 얻은 바닥 마스크는 색상·영역 후보를 보조할 뿐 서비스라인·3점선·골라인의 물리 보정을 대신하지 않는다.

## 점수판·VLM 보조

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)의 PP-OCRv5 계열을 점수판 crop의 초기 비교 후보로 둔다.
  전체 프레임 OCR 대신 작은 영역과 낮은 주기로 읽고 여러 시점의 일관성·지연을 확인한다.
  점수판이 없는 영상도 처리해야 한다. 한 번 읽은 점수 증가가 정확한 골/착지 프레임을 제공하지 않는다.
  Mac에서 사용할 런타임은 미검증이며 CPU OCR을 쓴다면 명시적인 보조 경로로 선택한다.
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)의 4B/8B는 소수 후보 프레임과 구조화된 관측을 검토할 비교 후보다.
  [4B-Instruct 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)를 기준으로 확인했다.
  이 모델이 경기 규칙을 자동 판정하거나 해당 Mac에서 충분히 빠르다고 확인한 것은 아니다.
  로컬 MPS/변환 경로·메모리·응답 시간은 별도 실험이 필요하다. 현재 코드에서 외부 API를 호출하지 않는다.
- Codex·Claude Code 등 에이전트도 기존 fusion 근거 묶음을 읽는 **검토자**로 연결할 수 있다.
  실제 존재하는 evidence ID만 인용하고, 흥미롭다는 판단을 서비스·착지·득점의 검증으로 승격시키지 않는다.

## 코드·가중치·데이터 사용 조건

조사한 저장소의 표시를 아래에 기록한다. 같은 논문이라도 구현·가중치·데이터의 조건은 다를 수 있으므로
실험에 선택한 체크포인트/배포판의 조건을 함께 저장한다.

| 항목 | 확인한 조건 / 출처 |
| --- | --- |
| SAM3 | 자체 [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). SAM2의 Apache-2.0으로 대체 해석하지 않음; 가중치 접근 승인 별도 |
| SAM2.1 / SAMURAI | SAM2 코드·체크포인트는 [Apache-2.0](https://github.com/facebookresearch/sam2#license), SAMURAI 코드는 [Apache-2.0](https://github.com/yangchris11/samurai/blob/master/LICENSE) |
| TrackNetV3 | [MIT](https://github.com/qaz812345/TrackNetV3/blob/master/LICENSE); 원문에서 사전 학습 체크포인트도 포함한다고 명시 |
| Ultralytics YOLO·YOLOE | [AGPL-3.0 / Enterprise](https://www.ultralytics.com/license). 모델·SDK를 프로젝트에 연결할 방식에 맞춰 조건 확인 |
| RTMPose / MMPose | 코드 [Apache-2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE); 모델 카드·학습 데이터 조건은 별도 확인 |
| ByteTrack / BoT-SORT | 원본 코드 각각 [MIT](https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE), [MIT](https://github.com/NirAharon/BoT-SORT/blob/main/LICENSE). Ultralytics에 포함된 구현을 원본 MIT로 일괄 해석하지 않음 |
| KaliCalib | 공식 저장소에 [CeCILL-2.1](https://github.com/DeepSportradar/2022-winners-camera-calibration-challenge) 표시 |
| PnLCalib | [코드 GPL-2.0-or-later / 사전 학습 가중치 CC BY 4.0](https://github.com/mguti97/PnLCalib/blob/main/LICENSE) |
| TVCalib | 코드 [MIT](https://github.com/MM4SPA/tvcalib/blob/main/LICENSE) |
| T-DEED / SoccerNet Team Spotting | 코드 [GPL-3.0](https://github.com/arturxe2/T-DEED/blob/main/LICENSE), [GPL-3.0](https://github.com/SoccerNet/sn-teamspotting/blob/main/LICENSE); SoccerNet 데이터 조건은 별도 |
| PaddleOCR / Qwen3-VL-4B-Instruct | 코드 [Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE), 해당 가중치 모델 카드 [Apache-2.0](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |

DeepSportRadar의 baseline·가중치·데이터와 SoccerNet Game State의 개별 구성 요소는 도입 시 각각 확인한다.
저장소에 접근 가능하거나 가중치 링크가 있다는 사실만으로 사용 조건이 모두 같은 것은 아니다.

## Mac 도입 순서와 검증

1. **기준선 보존:** 이미 실행한 YOLO11 Pose/객체·SAM3 이미지 MPS, OpenCV·CPU ByteTrack을 유지한다.
   현재 `sam.py`는 마스크를 저장하지 않는다. SAM3 비디오·SAMURAI·새 종목 모델은 조사 상태다.
2. **배드민턴:** 실제 대상 코트의 수동 landmark 정답부터 확보한다. TrackNetV3를 연속 구간에서 비교하되
   기존 포즈 후보 밖에도 독립 구간을 예약한다. 작은 움직임의 네트 플레이와 셔틀 줍기를 함께 평가한다.
3. **농구:** 대상 코트·실제 경기 골대를 고르고 공/림 검출을 만든다. KaliCalib와 선 기반 보정을 비교한다.
   림 주변 연속 공 관측 + ByteTrack 기준선 후, 가림이 남는 경우 SAM2.1/SAMURAI를 추가 비교한다.
4. **축구:** PnLCalib/TVCalib 보정, 카메라 구간, 선수/공 추적을 먼저 검증한다. 이후 T-DEED 이벤트 후보를
   추가한다. 중계 학습 모델의 동호회 영상 전이는 별도 성능 항목이다.
5. **보조 모델:** 필요할 때 RTMPose, YOLOE, BoT-SORT, OCR/VLM을 한 종류씩 비교한다.
   기존 후보의 재순위가 목적이면 이미 저장한 근거를 우선 재사용한다.

각 실험은 모델/가중치 hash·버전, 실제 백엔드·정밀도·입력 크기, 분석한 프레임/미분석 구간,
로드·디코딩·전송·새 추론·캐시 분석·인코딩 시간을 저장한다. 지원하지 않는 연산은 숨겨서 CPU로 보내지 않는다.
공식 CUDA FPS, Core ML 내보내기 지원, 체크포인트 공개는 Mac에서 빠르게 실행된다는 증거가 아니다.

정답 세트는 종목·구장·경기·카메라 단위로 나누고, 예전 포즈 후보 밖의 좋은 장면도 라벨링한다.
코트 선택/보정, 직접 관측한 공 위치, ID 전환, 이벤트 시점, 완결된 플레이, 사용자 하이라이트 선호를 따로 평가한다.
동일 Mac·입력·시간 예산에서 누락 감소와 오탐 증가를 함께 비교한다. Mac 자원 카운터는 다른 앱의 영향도 받고,
parent RSS만으로 FFmpeg·GPU 전체 메모리를 표현할 수 없다. 기존 벤치마크는 새 모델 조합의 성능 측정이 아니다.
