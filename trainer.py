"""
Training utilities and custom trainer
"""

import os
import logging
from typing import Dict, Any
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer
)
from datasets import Dataset
import torch

from config import training_config

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """커스텀 트레이너 클래스"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """손실 함수 계산 - 최신 transformers 버전 호환"""
        labels = inputs.get("labels")
        outputs = model(**inputs)

        # Shift so that tokens < n predict n
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss

class TrainingManager:
    """훈련 관리 클래스"""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = None

    def create_training_arguments(self, output_dir: str = None) -> TrainingArguments:
        """훈련 인자 생성"""
        if output_dir is None:
            output_dir = training_config.output_dir

        logger.info("Creating training arguments...")

        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,

            # 배치 크기 및 그래디언트 설정
            per_device_train_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,

            # 학습률 및 스케줄러
            learning_rate=training_config.learning_rate,
            lr_scheduler_type=training_config.lr_scheduler_type,
            warmup_ratio=training_config.warmup_ratio,

            # 에폭 및 스텝
            num_train_epochs=training_config.num_train_epochs,
            max_steps=training_config.max_steps,

            # 저장 설정
            save_strategy=training_config.save_strategy,
            save_total_limit=training_config.save_total_limit,

            # 로깅
            logging_steps=training_config.logging_steps,
            logging_strategy=training_config.logging_strategy,

            # 메모리 최적화 (그래디언트 체크포인팅 비활성화)
            dataloader_pin_memory=training_config.dataloader_pin_memory,
            gradient_checkpointing=False,  # 비활성화하여 그래디언트 문제 해결
            fp16=training_config.fp16,

            # 기타
            remove_unused_columns=training_config.remove_unused_columns,
            report_to=training_config.report_to,
            seed=training_config.seed,

            # 추가 최적화 설정
            dataloader_num_workers=0,  # 메모리 절약
            disable_tqdm=False,
            load_best_model_at_end=False,
        )

        logger.info(f"Training arguments created for output dir: {output_dir}")
        return args

    def create_data_collator(self) -> DataCollatorForLanguageModeling:
        """데이터 콜레이터 생성"""
        logger.info("Creating data collator...")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM이므로 False
            pad_to_multiple_of=8,  # 효율성을 위해
            return_tensors="pt"
        )

        return data_collator

    def create_trainer(self, tokenized_dataset: Dataset, output_dir: str = None) -> Trainer:
        """트레이너 생성"""
        logger.info("Creating trainer...")

        # 훈련 인자 생성
        training_args = self.create_training_arguments(output_dir)

        # 데이터 콜레이터 생성
        data_collator = self.create_data_collator()

        # 기본 Trainer 사용 (호환성 문제 해결)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )

        self.trainer = trainer
        logger.info("Trainer created successfully")
        return trainer

    def train(self, tokenized_dataset: Dataset, output_dir: str = None) -> Dict[str, Any]:
        """모델 훈련 실행"""
        logger.info("Starting training...")

        # 트레이너 생성 (없는 경우)
        if self.trainer is None:
            self.create_trainer(tokenized_dataset, output_dir)

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # 훈련 시작
            train_result = self.trainer.train()

            # 훈련 메트릭 로그
            logger.info("Training completed!")
            logger.info(f"Training loss: {train_result.training_loss:.4f}")
            logger.info(f"Training steps: {train_result.global_step}")

            return train_result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save_model(self, output_dir: str = None):
        """모델 저장"""
        if output_dir is None:
            output_dir = training_config.output_dir

        logger.info(f"Saving model to: {output_dir}")

        if self.trainer is None:
            raise ValueError("No trainer available for saving")

        # 모델 저장
        self.trainer.save_model(output_dir)

        # 토크나이저 저장
        self.tokenizer.save_pretrained(output_dir)

        logger.info("Model and tokenizer saved successfully")

    def get_training_stats(self) -> Dict[str, Any]:
        """훈련 통계 정보 반환"""
        if self.trainer is None:
            return {}

        stats = {
            "total_steps": self.trainer.state.global_step,
            "current_epoch": self.trainer.state.epoch,
            "learning_rate": self.trainer.get_lr()[0] if self.trainer.get_lr() else None,
        }

        # 로그 히스토리가 있는 경우
        if hasattr(self.trainer.state, 'log_history') and self.trainer.state.log_history:
            last_log = self.trainer.state.log_history[-1]
            stats.update({
                "last_loss": last_log.get("train_loss"),
                "last_learning_rate": last_log.get("learning_rate"),
            })

        return stats

def create_training_manager(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> TrainingManager:
    """TrainingManager 인스턴스 생성"""
    return TrainingManager(model, tokenizer)

def setup_training_environment():
    """훈련 환경 설정"""
    # 환경 변수 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 로깅 레벨 설정
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    logger.info("Training environment setup completed")

def estimate_training_time(dataset_size: int, batch_size: int = None, num_epochs: int = None) -> str:
    """예상 훈련 시간 계산"""
    if batch_size is None:
        batch_size = training_config.batch_size * training_config.gradient_accumulation_steps

    if num_epochs is None:
        num_epochs = training_config.num_train_epochs

    # 대략적인 추정 (RTX 4060 기준)
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * num_epochs

    # 1 step당 약 2-3초 추정
    estimated_seconds = total_steps * 2.5

    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)

    return f"약 {hours}시간 {minutes}분 (총 {total_steps} 스텝)"