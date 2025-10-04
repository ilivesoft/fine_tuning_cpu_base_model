"""
Model loading and management utilities
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Tuple

from config import model_config, lora_config

logger = logging.getLogger(__name__)

class ModelManager:
    """모델 로딩 및 관리 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            logger.info("Set pad_token to eos_token")
        
        self.tokenizer = tokenizer
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    
    def load_base_model(self, model_name: str = None) -> AutoModelForCausalLM:
        """베이스 모델 로드"""
        if model_name is None:
            model_name = model_config.model_name
            
        logger.info(f"Loading base model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        # BitsAndBytesConfig 설정 (8bit 양자화)
        quantization_config = None
        if model_config.use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                logger.info("8-bit quantization enabled")
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, loading without quantization")
                quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_config.torch_dtype,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            quantization_config=quantization_config
        )

        self.model = model

        # 모델 정보 출력
        total_params = model.num_parameters()
        logger.info(f"Model loaded successfully")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Model dtype: {model.dtype}")

        return model

    def setup_lora(self, model: AutoModelForCausalLM = None) -> AutoModelForCausalLM:
        """LoRA 설정"""
        if model is None:
            model = self.model

        logger.info("Setting up LoRA configuration...")

        # LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias="none"
        )

        # LoRA 모델 생성
        model = get_peft_model(model, peft_config)

        # 모델을 명시적으로 훈련 모드로 설정
        model.train()

        # LoRA 파라미터가 trainable인지 확인
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # 훈련 가능한 파라미터 정보 출력
        model.print_trainable_parameters()

        self.model = model
        logger.info("LoRA setup completed")

        return model

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """모델과 토크나이저를 함께 로드"""
        logger.info("Loading model and tokenizer...")

        # 토크나이저 로드
        tokenizer = self.load_tokenizer()

        # 모델 로드
        model = self.load_base_model()

        # LoRA 설정
        model = self.setup_lora(model)

        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer

    def save_model(self, output_dir: str, tokenizer: AutoTokenizer = None):
        """모델 저장"""
        logger.info(f"Saving model to: {output_dir}")

        if self.model is None:
            raise ValueError("No model to save")

        # 모델 저장
        self.model.save_pretrained(output_dir)

        # 토크나이저 저장
        if tokenizer is None:
            tokenizer = self.tokenizer

        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            logger.info("Model and tokenizer saved successfully")
        else:
            logger.warning("Tokenizer not provided for saving")

class InferenceModelManager:
    """추론용 모델 관리 클래스"""

    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = model_path
        self.base_model = base_model or model_config.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_finetuned_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Fine-tuned 모델 로드"""
        logger.info(f"Loading fine-tuned model from: {self.model_path}")
        logger.info(f"Base model: {self.base_model}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=model_config.trust_remote_code
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=model_config.torch_dtype,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True) if model_config.use_8bit else None
        )

        # Fine-tuned 모델 로드
        self.model = PeftModel.from_pretrained(base_model, self.model_path)

        logger.info(f"Fine-tuned model loaded on {self.device}")
        return self.model, self.tokenizer

def create_model_manager() -> ModelManager:
    """ModelManager 인스턴스 생성"""
    return ModelManager()

def create_inference_manager(model_path: str, base_model: str = None) -> InferenceModelManager:
    """InferenceModelManager 인스턴스 생성"""
    return InferenceModelManager(model_path, base_model)

def check_gpu_memory():
    """GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB

            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Allocated: {memory_allocated:.2f} GB")
            logger.info(f"  Reserved: {memory_reserved:.2f} GB")
            logger.info(f"  Total: {memory_total:.2f} GB")
    else:
        logger.info("CUDA not available")

def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")