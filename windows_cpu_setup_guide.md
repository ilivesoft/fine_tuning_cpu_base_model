# Windows 10 CPU 전용 Llama 파인튜닝 설정 가이드

## 🖥️ CPU 전용 시스템 요구사항

### 최소 요구사항
- **OS**: Windows 10 (64-bit)
- **CPU**: Intel i5-8세대 이상 또는 AMD Ryzen 5 3세대 이상
- **RAM**: 32GB 이상 **필수** (GPU VRAM 대신 사용)
- **저장공간**: 최소 100GB 여유 공간 (SSD 권장)

### 권장 요구사항
- **CPU**: Intel i7/i9 또는 AMD Ryzen 7/9 (멀티코어 고성능)
- **RAM**: 64GB 이상
- **저장공간**: NVMe SSD 200GB 이상

> ⚠️ **중요**: CPU 훈련은 GPU보다 20-50배 느립니다. 작은 데이터셋으로 테스트 후 진행하세요.

---

## 🔧 1단계: 기본 소프트웨어 설치

### 1.1 Python 설치
```cmd
# Python 3.11 다운로드 및 설치
# https://www.python.org/downloads/windows/
# "Add Python to PATH" 반드시 체크

# 설치 확인
python --version
pip --version
```

### 1.2 Git 설치 (선택사항)
- [Git for Windows](https://git-scm.com/download/win)

---

## 📦 2단계: CPU 전용 환경 설정

### 2.1 가상환경 생성
```cmd
# 프로젝트 폴더 생성
mkdir D:\Projects\llama_cpu_project
cd /d D:\Projects\llama_cpu_project

# 가상환경 생성
python -m venv llama_cpu_env

# 가상환경 활성화
llama_cpu_env\Scripts\activate
```

### 2.2 CPU 버전 PyTorch 설치
```cmd
# CPU 전용 PyTorch 설치 (CUDA 없음)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CPU only: {not torch.cuda.is_available()}')"
```

### 2.3 필수 라이브러리 설치
```cmd
# 핵심 라이브러리 (CPU 최적화)
pip install transformers>=4.53.0
pip install accelerate>=1.8.1
pip install peft>=0.16.0
pip install datasets>=3.6.0

# 데이터 처리
pip install pandas numpy pyarrow

# 유틸리티
pip install tqdm psutil
pip install tokenizers safetensors huggingface-hub
pip install Jinja2 PyYAML requests
```

> 📝 **주의**: `bitsandbytes`는 설치하지 않습니다 (GPU 전용)

---

## ⚙️ 3단계: CPU 전용 설정 파일 수정

### 3.1 `config_cpu.py` 생성
기존 `config.py` 대신 CPU 최적화 버전:

```python
"""
CPU 전용 설정 파일
"""

import torch
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """CPU 최적화 모델 설정"""
    # 더 작은 모델 사용 권장
    model_name: str = "microsoft/DialoGPT-small"  # 작은 모델로 변경
    max_length: int = 256  # 시퀀스 길이 단축
    torch_dtype: torch.dtype = torch.float32  # CPU는 float32 권장
    use_8bit: bool = False  # CPU에서는 불가능
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True

@dataclass
class LoRAConfig:
    """CPU 최적화 LoRA 설정"""
    r: int = 8  # rank 감소로 메모리 절약
    lora_alpha: int = 16  # 비례적으로 감소
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            # 더 적은 모듈 타겟팅
            self.target_modules = [
                "c_attn", "c_proj"  # DialoGPT용
            ]

@dataclass
class TrainingConfig:
    """CPU 최적화 훈련 설정"""
    csv_path: str = "civil_law_qa_dataset.csv"
    output_dir: str = "./fine_tuned_model_cpu"
    
    # CPU 최적화 배치 설정
    batch_size: int = 1  # 최소 배치 크기
    gradient_accumulation_steps: int = 32  # 효과적 배치 크기 = 32
    
    # 학습률 및 스케줄러
    learning_rate: float = 5e-5  # 더 낮은 학습률
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    
    # 에폭 설정 (CPU는 시간이 오래 걸림)
    num_train_epochs: int = 1  # 일단 1 에폭으로 테스트
    max_steps: int = 100  # 또는 제한된 스텝 수
    
    # 저장 설정
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    
    # 로깅
    logging_steps: int = 10
    logging_strategy: str = "steps"
    
    # CPU 최적화
    fp16: bool = False  # CPU는 FP16 미지원
    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 0  # CPU만 사용
    remove_unused_columns: bool = False
    
    # 기타
    seed: int = 42
    report_to: str = None

@dataclass
class InferenceConfig:
    """CPU 최적화 추론 설정"""
    max_new_tokens: int = 256  # 더 적은 토큰
    temperature: float = 0.8
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.1

@dataclass
class SystemConfig:
    """CPU 전용 시스템 설정"""
    device: str = "cpu"  # 강제로 CPU 사용
    cuda_visible_devices: str = ""  # GPU 비활성화
    pytorch_cuda_alloc_conf: str = ""
    tokenizers_parallelism: bool = True  # CPU는 병렬화 가능

# 전역 설정 인스턴스
model_config = ModelConfig()
lora_config = LoRAConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
system_config = SystemConfig()

def get_system_prompt() -> str:
    """시스템 프롬프트 반환"""
    return "당신은 한국 민법, 특히 부동산과 전세 관련 법률 전문가입니다. 정확하고 도움이 되는 법률 조언을 제공해주세요."

def format_prompt(question: str, answer: str = None) -> str:
    """간단한 프롬프트 포맷팅 (CPU 최적화)"""
    if answer:
        # 훈련용
        return f"질문: {question}\n답변: {answer}"
    else:
        # 추론용
        return f"질문: {question}\n답변:"
```

### 3.2 `model_manager_cpu.py` 생성
CPU 전용 모델 매니저:

```python
"""
CPU 전용 모델 관리
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Tuple

from config_cpu import model_config, lora_config

logger = logging.getLogger(__name__)

class CPUModelManager:
    """CPU 전용 모델 관리 클래스"""
    
    def __init__(self):
        # 강제로 CPU 사용
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        
    def load_tokenizer(self, model_name: str = None) -> AutoTokenizer:
        """토크나이저 로드"""
        if model_name is None:
            model_name = model_config.model_name
            
        logger.info(f"Loading tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=model_config.trust_remote_code,
            padding_side="right"
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.tokenizer = tokenizer
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    
    def load_base_model(self, model_name: str = None) -> AutoModelForCausalLM:
        """CPU 전용 베이스 모델 로드"""
        if model_name is None:
            model_name = model_config.model_name
            
        logger.info(f"Loading base model on CPU: {model_name}")
        
        # CPU 전용 로딩
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_config.torch_dtype,  # float32
            device_map={"": self.device},  # 명시적으로 CPU 지정
            trust_remote_code=model_config.trust_remote_code,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        )
        
        # CPU로 이동
        model = model.to(self.device)
        
        self.model = model
        
        # 모델 정보 출력
        total_params = model.num_parameters()
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Model dtype: {model.dtype}")
        
        return model
    
    def setup_lora(self, model: AutoModelForCausalLM = None) -> AutoModelForCausalLM:
        """CPU 최적화 LoRA 설정"""
        if model is None:
            model = self.model
            
        logger.info("Setting up LoRA for CPU...")
        
        # CPU 최적화 LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.r,  # 8로 감소
            lora_alpha=lora_config.lora_alpha,  # 16으로 감소
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias="none"
        )
        
        # LoRA 모델 생성
        model = get_peft_model(model, peft_config)
        model = model.to(self.device)  # CPU로 확실히 이동
        
        # 훈련 모드로 설정
        model.train()
        
        # 훈련 가능한 파라미터 정보 출력
        model.print_trainable_parameters()
        
        self.model = model
        logger.info("LoRA setup completed for CPU")
        
        return model
    
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """모델과 토크나이저를 함께 로드"""
        logger.info("Loading model and tokenizer for CPU...")
        
        # 토크나이저 로드
        tokenizer = self.load_tokenizer()
        
        # 모델 로드
        model = self.load_base_model()
        
        # LoRA 설정
        model = self.setup_lora(model)
        
        logger.info("Model and tokenizer loaded successfully on CPU")
        return model, tokenizer

def create_cpu_model_manager() -> CPUModelManager:
    """CPUModelManager 인스턴스 생성"""
    return CPUModelManager()

def check_cpu_info():
    """CPU 정보 확인"""
    import psutil
    
    cpu_count = psutil.cpu_count(logical=False)  # 물리적 코어
    cpu_count_logical = psutil.cpu_count(logical=True)  # 논리적 코어
    memory_gb = psutil.virtual_memory().total / 1024**3
    
    logger.info(f"Physical CPU cores: {cpu_count}")
    logger.info(f"Logical CPU cores: {cpu_count_logical}")
    logger.info(f"Total RAM: {memory_gb:.1f} GB")
    logger.info(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # PyTorch 멀티스레딩 설정
    torch.set_num_threads(cpu_count)  # 물리적 코어 수로 설정
    logger.info(f"PyTorch threads set to: {torch.get_num_threads()}")
```

---

## 🚀 4단계: CPU 전용 실행 스크립트

### 4.1 `run_cpu_scripts.bat` 생성

```batch
@echo off
setlocal enabledelayedexpansion

REM CPU 전용 환경 변수 설정
set CUDA_VISIBLE_DEVICES=
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set TOKENIZERS_PARALLELISM=true

echo [94m================================[0m
echo [94m   CPU 전용 Llama Fine-tuning  [0m
echo [94m================================[0m

if "%1"=="" goto :show_help
if "%1"=="help" goto :show_help
if "%1"=="check" goto :check_system
if "%1"=="train_small" goto :train_small
if "%1"=="train_test" goto :train_test
if "%1"=="chat" goto :chat
if "%1"=="test" goto :test

:show_help
echo.
echo [93m사용법:[0m
echo   %0 [명령]
echo.
echo [93m명령어:[0m
echo   check       - 시스템 정보 확인
echo   train_test  - 테스트 훈련 (10 스텝)
echo   train_small - 소규모 훈련 (100 스텝)
echo   chat        - 대화형 추론
echo   test        - 빠른 테스트
echo   help        - 도움말
echo.
echo [91m주의: CPU 훈련은 매우 느립니다![0m
goto :end

:check_system
echo [96m시스템 정보 확인 중...[0m
python -c "import torch, psutil; print(f'CPU cores: {psutil.cpu_count()}'); print(f'RAM: {psutil.virtual_memory().total/1024**3:.1f}GB'); print(f'PyTorch CPU only: {not torch.cuda.is_available()}')"
goto :end

:train_test
echo [92m테스트 훈련 시작 (10 스텝)...[0m
python main_train_cpu.py --max_steps 10 --save_steps 5
goto :end

:train_small
echo [92m소규모 훈련 시작 (100 스텝)...[0m
python main_train_cpu.py --max_steps 100 --save_steps 25
goto :end

:chat
echo [92m대화형 추론 시작...[0m
python main_inference_cpu.py --model_path ./fine_tuned_model_cpu --interactive
goto :end

:test
echo [92m빠른 테스트 시작...[0m
python main_inference_cpu.py --model_path ./fine_tuned_model_cpu --quick_test
goto :end

:end
pause
```

---

## 📝 5단계: CPU 최적화된 메인 스크립트 수정

### 5.1 주요 수정사항

#### `main_train_cpu.py` (main_train.py 수정 버전)
```python
# 맨 위에 CPU 설정 추가
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU 비활성화

# config.py 대신 config_cpu.py import
from config_cpu import training_config, model_config
from model_manager_cpu import create_cpu_model_manager, check_cpu_info
```

#### 추가 최적화 설정
```python
# CPU 멀티스레딩 최적화
import torch
torch.set_num_threads(8)  # CPU 코어 수에 맞게 조정

# 메모리 효율성을 위한 설정
torch.backends.cudnn.enabled = False  # CUDNN 비활성화
```

---

## ⚠️ 6단계: CPU 훈련 주의사항 및 최적화

### 6.1 현실적인 기대치 설정

```cmd
# 🐌 속도 비교 (참고용)
# GPU (RTX 4060): 1 epoch ≈ 30분
# CPU (i7-10세대): 1 epoch ≈ 10-20시간

# 따라서 처음에는 매우 작은 데이터셋으로 테스트
```

### 6.2 메모리 최적화 설정

#### Windows 가상 메모리 설정
1. **제어판** → **시스템** → **고급 시스템 설정**
2. **성능** → **설정** → **고급** → **가상 메모리**
3. **사용자 지정 크기**: 초기값 16GB, 최대값 32GB 설정

#### 시스템 최적화
```cmd
# 백그라운드 앱 종료
taskkill /f /im chrome.exe
taskkill /f /im firefox.exe

# 고성능 전원 모드 설정
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

### 6.3 작은 데이터셋으로 테스트

#### `test_dataset.csv` 생성 (10개 샘플만)
```csv
question,answer,category,difficulty
"전세권이란 무엇인가요?","전세권은 전세금을 지급하고 타인의 부동산을 점유하여 사용·수익할 수 있는 물권입니다.","부동산물권","초급"
"전세 사기 예방법은?","전세권 등기, 확정일자, 전세보증보험 가입 등이 있습니다.","전세사기","초급"
```

---

## 🚀 7단계: 실행 순서

### 7.1 단계별 테스트 실행

```cmd
# 1. 가상환경 활성화
llama_cpu_env\Scripts\activate

# 2. CPU 정보 확인
run_cpu_scripts.bat check

# 3. 매우 작은 테스트 (10 스텝)
run_cpu_scripts.bat train_test

# 4. 결과 확인
run_cpu_scripts.bat test

# 5. 성공하면 더 큰 훈련
run_cpu_scripts.bat train_small
```

### 7.2 성능 모니터링

```cmd
# 작업 관리자에서 모니터링
# - CPU 사용률: 90-100% 유지되어야 함
# - RAM 사용률: 80% 이하 유지
# - 온도: CPU 온도 확인 (과열 주의)
```

---

## 💡 추가 최적화 팁

### CPU 성능 극대화
1. **BIOS 설정**: Turbo Boost/Precision Boost 활성화
2. **전원 관리**: 고성능 모드
3. **백그라운드 프로세스**: 최소화
4. **쿨링**: CPU 온도 65°C 이하 유지

### 더 작은 모델 사용
```python
# config_cpu.py에서 더 작은 모델로 변경
model_name: str = "microsoft/DialoGPT-small"  # 117M 파라미터
# 또는
model_name: str = "distilgpt2"  # 82M 파라미터
```

### 점진적 학습
```python
# 매우 작은 학습률과 적은 스텝으로 시작
learning_rate: float = 1e-5
max_steps: int = 50
```

---

## ⚡ 성능 예상

| 설정 | 예상 시간 (i7 8세대 기준) | 권장 사용 |
|------|------------------------|-----------|
| 10 스텝 테스트 | 5-10분 | 초기 테스트 |
| 100 스텝 | 30-60분 | 기능 확인 |
| 1 에폭 (전체) | 10-20시간 | 실제 훈련 |

CPU 훈련은 느리지만 가능합니다! 작은 데이터셋으로 시작해서 점진적으로 확장하는 것이 좋습니다. 💪