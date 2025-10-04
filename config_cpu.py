"""
Windows CPU 전용 설정 파일
"""

import torch
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """CPU 최적화 모델 설정"""
    # 작은 모델 사용 (CPU에서 실행 가능)
    model_name: str = "skt/kogpt2-base-v2"  # 한국어 GPT2 소형 모델
    max_length: int = 256  # 시퀀스 길이 단축
    torch_dtype: torch.dtype = torch.float32  # CPU는 float32
    use_8bit: bool = False  # CPU에서 불가능
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True

@dataclass
class LoRAConfig:
    """CPU 최적화 LoRA 설정"""
    r: int = 8  # rank 감소
    lora_alpha: int = 16  # 비례 감소
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            # GPT2 아키텍처용 타겟 모듈
            self.target_modules = [
                "c_attn", "c_proj"
            ]

@dataclass
class TrainingConfig:
    """CPU 최적화 훈련 설정"""
    csv_path: str = "civil_law_qa_dataset.csv"
    output_dir: str = "./fine_tuned_model_cpu"
    
    # CPU 최적화 배치 설정
    batch_size: int = 1  # 최소 배치
    gradient_accumulation_steps: int = 32  # 효과적 배치 = 32
    
    # 학습률
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    
    # 에폭 (CPU는 느리므로 제한적으로)
    num_train_epochs: int = 1
    max_steps: int = 100  # 기본 100 스텝
    
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
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    
    seed: int = 42
    report_to: str = None

@dataclass
class InferenceConfig:
    """CPU 최적화 추론 설정"""
    max_new_tokens: int = 256
    temperature: float = 0.8
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.1

@dataclass
class SystemConfig:
    """CPU 전용 시스템 설정"""
    device: str = "cpu"
    cuda_visible_devices: str = ""  # GPU 비활성화
    pytorch_cuda_alloc_conf: str = ""
    tokenizers_parallelism: bool = True

# 전역 설정 인스턴스
model_config = ModelConfig()
lora_config = LoRAConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
system_config = SystemConfig()

def get_system_prompt() -> str:
    """시스템 프롬프트"""
    return "당신은 한국 민법, 특히 부동산과 전세 관련 법률 전문가입니다. 정확하고 도움이 되는 법률 조언을 제공해주세요."

def format_prompt(question: str, answer: str = None) -> str:
    """간단한 프롬프트 포맷팅"""
    if answer:
        return f"질문: {question}\n답변: {answer}"
    else:
        return f"질문: {question}\n답변:"
