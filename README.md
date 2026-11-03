# A6-1 - AI 엔진을 직접 만들기 (numpy-autograd)

> 🚧 **미착수** — 디렉터리·레포 준비만 완료

NumPy 만으로 Tensor·AutoGrad·레이어·옵티마이저를 갖춘 미니 딥러닝 프레임워크를 만든다.
「구현 → 검증 → 실험」 순서로 진행한다. Gradient Checking 으로 역전파가 맞는지 증명하고,
Zero/Random/He 초기화 실험으로 학습이 실패하는 상황을 재현해 원인을 분석한다.

- 과제 원문: 로컬 `../a6-1-numpy-autograd.md`
- AI 심화 학습 · 딥러닝 · 80시간 · 필수 · 개인
- Python 3.10 이상 권장
- 선행 권장: A2-1 (역전파 손계산·중심차분 수치 미분·Momentum 이 여기서 코드로 다시 나온다)

## 설치 및 실행

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt   # uv 가 없으면 python -m venv .venv 후 pip install -r requirements.txt
```

진입점(`gradient_check.py`, XOR·MNIST 학습 스크립트)은 착수 때 정한다.

## 라이브러리 제약 (과제 §7)

| 구분 | 내용 |
|---|---|
| 구현 | **NumPy 만** — PyTorch·TensorFlow 는 로직 참고용으로만 |
| 데이터 로더 | `torch.utils.data` 등 자동 로더 금지, 직접 구현 |
| MNIST | 바이너리(UBYTE) 또는 CSV 를 직접 파싱 (다운로드에는 requests·urllib·gzip 허용) |

## 성공 판정 최소 기준 (과제 §4)

| 항목 | 기준 |
|---|---|
| Gradient Check | 모든 레이어 상대 오차 < 1e-7 (ε = 1e-5 중심차분) |
| XOR (필수) | He Init, 100 Epoch 안에 Loss < 0.1 또는 정확도 95% 이상 |
| MNIST (선택) | 1 Epoch 정확도 80% 이상 |
| Zero Init 실패 재현 | 50 Epoch 후에도 Loss 감소 없음 |

## 제출물 체크리스트

- [ ] 프레임워크 — tensor·autograd·layer·optimizer 모듈 (사칙연산·MatMul 순전파/역전파, Linear·ReLU·Sigmoid·Softmax, 초기화 Zero/Random/He/Xavier, SGD·Adam)
- [ ] 검증 리포트 — `gradient_check.py` 실행 로그
- [ ] 실험 리포트 — Zero vs Random vs He Loss 비교 그래프(`figures/`) + 분석 (Xavier 는 구현 필수·비교 선택)
- [ ] 학습 검증 스크립트 — XOR 해결 + MNIST 분류
- [ ] README — 구조(core/nn/optim 의존 관계)·설치·실행 순서·Tensor → 연산 → backward 사용 예시·실험 요약

## 보너스

- [ ] Inverted Dropout + L2 Regularization(Weight Decay)
- [ ] 추가 활성화 함수 (Tanh·LeakyReLU·ELU) 역전파
