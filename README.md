# 🏛️ 민법 부동산 법률 상담 챗봇 (GPT2 Fine-tuning)

한국어 GPT2 모델을 활용한 민법 부동산 및 전세 관련 법률 상담 챗봇입니다. LoRA(Low-Rank Adaptation)를 사용하여 효율적으로 fine-tuning하며, CPU 및 GPU 환경 모두를 지원합니다.

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [시스템 요구사항](#-시스템-요구사항)
- [설치 방법](#-설치-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [사용 방법](#-사용-방법)
- [설정 파일 설명](#-설정-파일-설명)
- [데이터셋 형식](#-데이터셋-형식)
- [트러블슈팅](#-트러블슈팅)
- [라이선스](#-라이선스)

## 🎯 프로젝트 개요

이 프로젝트는 한국 민법, 특히 부동산 및 전세 관련 법률 질문에 답변하는 AI 챗봇을 구축합니다. **Llama 모델에서 GPT2 모델로 변경**하여 더 가벼운 환경에서도 실행 가능하도록 최적화했습니다.

### 주요 변경사항 (Llama → GPT2)

- **모델**: Meta Llama → skt/kogpt2-base-v2
- **아키텍처**: Llama → GPT2
- **타겟 모듈**: `q_proj`, `v_proj` → `c_attn`, `c_proj`
- **프롬프트 형식**: 복잡한 chat template → 간단한 Q&A 형식
- **토큰 처리**: 특수 토큰 간소화

## ✨ 주요 특징

- ✅ **경량 모델**: skt/kogpt2-base-v2 사용으로 리소스 효율성 향상
- ✅ **LoRA Fine-tuning**: 전체 모델 재학습 대신 일부 파라미터만 학습
- ✅ **CPU/GPU 지원**: CPU 전용 환경에서도 훈련 및 추론 가능
- ✅ **대화형 인터페이스**: 실시간 질의응답 기능
- ✅ **메모리 최적화**: 8bit 양자화 및 그래디언트 누적 지원
- ✅ **모듈화된 구조**: 재사용 가능한 컴포넌트 설계

## 💻 시스템 요구사항

### GPU 환경
- **GPU**: NVIDIA RTX 4060 8GB 이상
- **RAM**: 16GB 이상
- **디스크**: 20GB 이상 여유 공간
- **CUDA**: 12.6 이상
- **Python**: 3.9 이상

### CPU 환경
- **CPU**: 멀티코어 프로세서 (4코어 이상 권장)
- **RAM**: 16GB 이상 (32GB 권장)
- **디스크**: 20GB 이상 여유 공간
- **Python**: 3.9

⚠️ **CPU 환경 주의사항**: CPU에서는 훈련 속도가 매우 느립니다 (GPU 대비 10~50배). 테스트 및 추론 용도로 권장합니다.

## 📦 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/duck3244/fine_tuning_cpu_base_model.git
```

### 2. 가상환경 생성 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

 - Miniconda 3 기준
  -- C:\Users\User\miniconda3>cd Scripts
  -- conda create -n py39_tf python=3.9
  -- conda activate py39_tf

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

#### GPU 환경

```bash
# PyTorch GPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 기타 패키지 설치
pip install -r requirements.txt
```

#### CPU 환경 (Windows) - 권장

```bash
# PyTorch CPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CPU 전용 패키지 설치
pip install -r requirements_cpu_py39.txt
```

### 4. 설치 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 프로젝트 구조

```
llama_finetune_project/
├── 📄 config_cpu.py              # CPU 전용 설정 파일 (GPT2)
├── 📄 data_loader.py             # 데이터 로딩 및 전처리
├── 📄 model_manager_cpu.py       # CPU 전용 모델 관리
├── 📄 trainer.py                 # 훈련 관련 유틸리티
├── 📄 inference_manager.py       # 추론 및 텍스트 생성
├── 📄 utils.py                   # 공통 유틸리티 함수
├── 📄 main_train_cpu.py          # CPU 전용 훈련 스크립트
├── 📄 main_inference_cpu.py      # CPU 전용 추론 스크립트
├── 📄 requirements.txt           # GPU 패키지 의존성
├── 📄 requirements_cpu_py39.txt  # CPU 패키지 의존성 (Python 3.9)
├── 📄 civil_law_qa_dataset.csv   # Q/A 데이터셋
├── 📄 README.md                  # 프로젝트 설명서
└── 📁 fine_tuned_model_cpu/      # 훈련된 모델 저장 디렉토리
```

## 🚀 사용 방법

### 1. 데이터 준비

`civil_law_qa_dataset.csv` 파일이 프로젝트 루트에 있는지 확인합니다.

```bash
# CSV 파일 검증
python -c "from utils import validate_csv_file; validate_csv_file('civil_law_qa_dataset.csv')"
```

### 2. 모델 훈련

#### CPU 환경에서 훈련

```bash
python main_train_cpu.py \
    --csv_path civil_law_qa_dataset.csv \
    --output_dir ./fine_tuned_model_cpu \
    --max_steps 100 \
    --batch_size 1 \
    --learning_rate 5e-5
```

#### 주요 훈련 파라미터

- `--csv_path`: 데이터셋 CSV 파일 경로
- `--output_dir`: 모델 저장 디렉토리
- `--max_steps`: 최대 훈련 스텝 수 (CPU는 100 권장)
- `--batch_size`: 배치 크기 (CPU는 1 권장)
- `--epochs`: 에폭 수 (CPU는 1 권장)
- `--learning_rate`: 학습률 (기본값: 5e-5)

#### 설정 검증만 실행 (Dry Run)

```bash
python main_train_cpu.py --dry_run
```

### 3. 모델 추론

#### 대화형 모드

```bash
python main_inference_cpu.py \
    --model_path ./fine_tuned_model_cpu \
    --interactive
```

#### 단일 질문

```bash
python main_inference_cpu.py \
    --model_path ./fine_tuned_model_cpu \
    --question "전세권이란 무엇인가요?"
```

#### 빠른 테스트

```bash
python main_inference_cpu.py \
    --model_path ./fine_tuned_model_cpu \
    --quick_test
```

#### 주요 추론 파라미터

- `--model_path`: Fine-tuned 모델 경로
- `--interactive`: 대화형 모드 실행
- `--question`: 단일 질문 입력
- `--quick_test`: 빠른 테스트 실행
- `--max_new_tokens`: 최대 생성 토큰 수 (기본값: 256)
- `--temperature`: 생성 온도 (기본값: 0.8)

## ⚙️ 설정 파일 설명

### config_cpu.py

GPT2 모델용 CPU 최적화 설정 파일입니다.

```python
# 모델 설정
model_name = "skt/kogpt2-base-v2"  # 한국어 GPT2
max_length = 256                    # 시퀀스 길이
torch_dtype = torch.float32         # CPU는 float32

# LoRA 설정
r = 8                               # LoRA rank (감소)
lora_alpha = 16                     # 스케일링 팩터
target_modules = ["c_attn", "c_proj"]  # GPT2 타겟 모듈

# 훈련 설정
batch_size = 1                      # 최소 배치
gradient_accumulation_steps = 32    # 효과적 배치 = 32
max_steps = 100                     # CPU 최적화
```

### 프롬프트 형식 변경

**이전 (Llama)**:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 한국 민법 전문가입니다.
<|start_header_id|>user<|end_header_id|>
질문내용
<|start_header_id|>assistant<|end_header_id|>
답변내용
```

**현재 (GPT2)**:
```
질문: 질문내용
답변: 답변내용</s>
```

## 📊 데이터셋 형식

CSV 파일은 다음 컬럼을 포함해야 합니다:

| 컬럼 | 필수 | 설명 |
|------|------|------|
| question | ✅ | 질문 내용 |
| answer | ✅ | 답변 내용 |
| category | ❌ | 카테고리 (선택) |
| difficulty | ❌ | 난이도 (선택) |

### 예시

```csv
question,answer,category,difficulty
"전세권이란 무엇인가요?","전세권은 전세금을 지급하고 타인의 부동산을 점유하여 그 부동산의 용도에 맞게 사용·수익하며, 후에 그 부동산 전부에 대하여 후순위권리자 기타 채권자보다 전세금의 우선변제를 받을 수 있는 권리입니다.",부동산물권,초급
```

## 🔧 트러블슈팅

### 1. GPU 메모리 부족

```bash
# config_cpu.py에서 배치 크기 줄이기
batch_size = 1
gradient_accumulation_steps = 32
```

### 2. CPU 훈련이 너무 느림

```bash
# max_steps 제한
python main_train_cpu.py --max_steps 50
```

### 3. 모듈을 찾을 수 없음 오류

```bash
# 패키지 재설치
pip install --upgrade -r requirements_cpu_py39.txt
```

### 4. 토크나이저 오류

GPT2 모델은 특별한 토큰 설정이 간단합니다:
- `pad_token = eos_token`
- `bos_token = eos_token`

### 5. 추론 결과가 이상함

```bash
# 생성 파라미터 조정
python main_inference_cpu.py \
    --temperature 0.7 \
    --max_new_tokens 512
```

## 📈 성능 최적화 팁

### CPU 환경

1. **스레드 수 최적화**: `config_cpu.py`에서 자동 설정됨
2. **배치 크기 최소화**: `batch_size=1`
3. **그래디언트 누적**: `gradient_accumulation_steps=32`
4. **스텝 수 제한**: `max_steps=100`

### 메모리 절약

1. **시퀀스 길이 감소**: `max_length=256`
2. **LoRA rank 감소**: `r=8`
3. **불필요한 로그 제거**: `report_to=None`

## 🎓 모델 정보

### GPT2 vs Llama 비교

| 특징 | GPT2 | Llama |
|------|------|-------|
| 모델 크기 | ~500MB | ~13GB |
| 메모리 요구량 | 낮음 | 높음 |
| 훈련 속도 | 빠름 | 느림 |
| 한국어 성능 | 우수 (skt/kogpt2) | 우수 |
| 추론 속도 | 매우 빠름 | 느림 |

### LoRA 파라미터

- **rank (r)**: 8 (Llama: 16)
- **alpha**: 16 (Llama: 32)
- **타겟 모듈**: `c_attn`, `c_proj` (GPT2 attention 레이어)

## 📝 로그 파일

훈련 및 추론 시 자동으로 로그 파일이 생성됩니다:

- `training_cpu_YYYYMMDD_HHMMSS.log`
- `inference_cpu_YYYYMMDD_HHMMSS.log`

---
