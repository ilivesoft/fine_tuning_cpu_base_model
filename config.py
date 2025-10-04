"""
Configuration settings for Llama fine-tuning project
"""

import torch
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """모델 관련 설정"""
    model_name: str = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"  # 요청한 모델
    max_length: int = 512
    torch_dtype: torch.dtype = torch.float16
    use_8bit: bool = False  # 8bit 양자화 비활성화 (그래디언트 문제 해결)
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True

@dataclass
class LoRAConfig:
    """LoRA 설정"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    # 데이터
    csv_path: str = "civil_law_qa_dataset.csv"
    output_dir: str = "./fine_tuned_model"

    # 배치 및 메모리 (8bit 양자화 없이도 안정적으로)
    batch_size: int = 2  # 배치 크기 줄임
    gradient_accumulation_steps: int = 8  # 그래디언트 누적 증가로 효과적 배치 크기 유지

    # 학습률 및 스케줄러
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # 에폭 및 스텝
    num_train_epochs: int = 3
    max_steps: int = -1

    # 저장 설정
    save_strategy: str = "epoch"
    save_total_limit: int = 2

    # 로깅
    logging_steps: int = 5
    logging_strategy: str = "steps"

    # 최적화
    fp16: bool = True
    gradient_checkpointing: bool = False  # 비활성화
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False

    # 기타
    seed: int = 42
    report_to: str = None  # wandb 등 비활성화

@dataclass
class InferenceConfig:
    """추론 관련 설정"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.1

@dataclass
class SystemConfig:
    """시스템 관련 설정"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_visible_devices: str = "0"
    pytorch_cuda_alloc_conf: str = "max_split_size_mb:512"
    tokenizers_parallelism: bool = False

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
    """프롬프트 포맷팅"""
    if answer:
        # 훈련용 프롬프트
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{get_system_prompt()}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""
    else:
        # 추론용 프롬프트
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{get_system_prompt()}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""