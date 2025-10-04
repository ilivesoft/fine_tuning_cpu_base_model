"""
Windows CPU 전용 모델 관리
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Tuple
import psutil

from config_cpu import model_config, lora_config

logger = logging.getLogger(__name__)

class CPUModelManager:
    """CPU 전용 모델 관리"""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        
        # CPU 멀티스레딩 최적화
        cpu_count = psutil.cpu_count(logical=False)
        torch.set_num_threads(cpu_count)
        logger.info(f"PyTorch threads set to: {cpu_count}")
        
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
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.tokenizer = tokenizer
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    
    def load_base_model(self, model_name: str = None) -> AutoModelForCausalLM:
        """CPU 전용 베이스 모델 로드"""
        if model_name is None:
            model_name = model_config.model_name
            
        logger.info(f"Loading model on CPU: {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_config.torch_dtype,
            device_map={"": self.device},
            trust_remote_code=model_config.trust_remote_code,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        )
        
        model = model.to(self.device)
        self.model = model
        
        total_params = model.num_parameters()
        logger.info(f"Model loaded on CPU")
        logger.info(f"Total parameters: {total_params:,}")
        
        return model
    
    def setup_lora(self, model: AutoModelForCausalLM = None) -> AutoModelForCausalLM:
        """LoRA 설정"""
        if model is None:
            model = self.model
            
        logger.info("Setting up LoRA for CPU...")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model = model.to(self.device)
        model.train()
        
        # LoRA 파라미터 확인
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        model.print_trainable_parameters()
        
        self.model = model
        logger.info("LoRA setup completed")
        
        return model
    
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """모델과 토크나이저 로드"""
        logger.info("Loading model and tokenizer for CPU...")
        
        tokenizer = self.load_tokenizer()
        model = self.load_base_model()
        model = self.setup_lora(model)
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    def save_model(self, output_dir: str, tokenizer: AutoTokenizer = None):
        """모델 저장"""
        logger.info(f"Saving model to: {output_dir}")
        
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_pretrained(output_dir)
        
        if tokenizer is None:
            tokenizer = self.tokenizer
            
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            logger.info("Model and tokenizer saved")

class InferenceModelManager:
    """추론용 모델 관리"""
    
    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = model_path
        self.base_model = base_model or model_config.model_name
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        
    def load_finetuned_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Fine-tuned 모델 로드"""
        logger.info(f"Loading fine-tuned model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=model_config.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=model_config.torch_dtype,
            device_map={"": self.device},
            trust_remote_code=model_config.trust_remote_code,
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded on CPU")
        return self.model, self.tokenizer

def create_cpu_model_manager() -> CPUModelManager:
    """CPUModelManager 생성"""
    return CPUModelManager()

def create_inference_manager(model_path: str, base_model: str = None) -> InferenceModelManager:
    """InferenceModelManager 생성"""
    return InferenceModelManager(model_path, base_model)

def check_cpu_memory():
    """CPU 메모리 확인"""
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / 1024**3:.1f} GB")
    logger.info(f"Available RAM: {memory.available / 1024**3:.1f} GB")
    logger.info(f"Used RAM: {memory.used / 1024**3:.1f} GB")
    logger.info(f"RAM Usage: {memory.percent}%")
