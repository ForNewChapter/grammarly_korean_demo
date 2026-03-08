# Edit Tagger v2

## 목표

현재 태거의 핵심 문제는 두 가지입니다.

1. 열어야 할 토큰을 `KEEP`으로 둔다.
- 예: `낫느냐에`

2. 건드리면 안 되는 토큰을 잘못 건드린다.
- 예: `다르다 -> 다른다`

즉 v2의 목적은 단순히 `non_keep_f1`을 올리는 것이 아니라,

- `OPEN_REPLACE recall`을 올리고
- `false positive`를 줄이며
- `PUNCT_FIX` 같은 애매한 fallback 라벨을 제거하는 것입니다.

## 라벨 설계

v1 라벨은 다음 문제가 있었습니다.

- `PUNCT_FIX`가 너무 넓다
- 애매한 오류가 `PUNCT_FIX`로 새기 쉽다
- 태거가 문장부호 교정까지 떠안고 있다

v2는 태거의 역할을 줄입니다.

- `KEEP`
- `SPACE_FIX`
- `OPEN_REPLACE`

설계 원리:

- 띄어쓰기는 태거가 열어도 괜찮다
- 문장부호는 규칙 단계가 처리한다
- 태거는 "치환 후보를 열어야 하는가"에 집중한다

## 데이터 구성

기본 데이터:

- AI Hub 정규화 train: `214,300`
- AI Hub 정규화 validation: `29,050`

v2 데이터셋은 두 종류의 예시를 같이 만든다.

1. noisy example
- `err_sentence`
- 오류 토큰에 `SPACE_FIX` 또는 `OPEN_REPLACE`

2. clean keep example
- `cor_sentence`
- 모든 토큰을 `KEEP`

이 clean 예시가 중요한 이유:

- `다르다`, `맞다`, `있다` 같은 정상 토큰을 괜히 건드리는 오탐을 줄인다

## 라벨 규칙

`SPACE_FIX`

- 띄어쓰기 태그가 있거나
- err/cor가 공백만 다를 때

`OPEN_REPLACE`

- 그 외의 철자/형태/기본형 치환

`KEEP`

- clean example 전체
- noisy example에서 오류가 아닌 나머지 토큰
- 문장부호만 바뀌는 경우는 태거 라벨에서는 `KEEP`

## 왜 이 구조가 맞는가

현재 파이프라인은 이미 아래처럼 나뉘어 있습니다.

1. 규칙 교정
2. 띄어쓰기
3. 태거
4. 후보 생성
5. 재정렬

그러므로 태거는 후보 생성기로 넘길 토큰만 잘 찾으면 됩니다.

태거가 너무 많은 라벨을 떠안으면

- `PUNCT_FIX` 같은 애매한 fallback이 생기고
- recall과 precision이 둘 다 망가집니다

## 추천 베이스 모델

1. `monologg/koelectra-base-v3-discriminator`
- 현재 코드와 가장 잘 맞다
- 첫 주력 모델

2. `beomi/KcELECTRA-base`
- noisy text 비교군
- SNS/댓글체 강점 가능성 확인용

3. `monologg/koelectra-small-v3-discriminator`
- student/온디바이스용

## 학습 전략

1. v2 데이터셋 생성
2. `KoELECTRA-base-v3`로 전체 학습
3. hard negative와 valid-to-valid family를 추가로 넣어 2차 학습
4. 필요하면 small 모델 distillation

권장 초기값:

- `max_length = 96`
- `batch_size = 8`
- `gradient_accumulation = 4`
- `learning_rate = 2e-5`
- `epochs = 2~3`

## 필수 지표

`accuracy`만 보면 안 됩니다.

반드시 봐야 할 지표:

- `non_keep_f1`
- `open_replace_precision`
- `open_replace_recall`
- `space_fix_precision`
- `space_fix_recall`
- `keep_false_positive_rate`

## 기대 개선

파인튜닝을 다시 하면 좋아질 가능성이 큰 부분:

1. `낫느냐에`, `좋겟다`를 더 자주 `OPEN_REPLACE`로 연다
2. `다르다 -> 다른다` 같은 오탐을 줄인다
3. `PUNCT_FIX`로 새는 현상을 없앤다

## 한계

태거만 좋아져도 최종 교정이 완벽해지는 것은 아닙니다.

예:

- `나았다 / 낳았다`

이런 문제는 태거가 토큰을 열어 주는 데는 도움이 되지만,
최종 선택은 여전히 후보 생성기와 문맥 점수가 필요합니다.

즉 v2는 다음을 노립니다.

- 후보 생성기로 넘길 span을 더 잘 찾는다
- 쓸데없는 span은 덜 연다

후보 선택 문제 자체를 태거만으로 해결하려 하지는 않습니다.
