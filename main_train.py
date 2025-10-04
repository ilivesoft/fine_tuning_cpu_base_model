#!/usr/bin/env python3
"""
메인 훈련 스크립트
Llama-3.2-Korean Fine-tuning for Civil Law Real Estate QA
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# 프로젝트 모듈 임포트
from config import training_config, model_config
from utils import (
    setup_logging,
    check_system_requirements,
    print_system_info,
    set_environment_variables,
    validate_csv_file,
    create_directory,
    clear_gpu_memory,
    estimate_model_size,
    ProgressTracker
)
from model_manager import create_model_manager, check_gpu_memory
from data_loader import load_and_prepare_data
from trainer import create_training_manager, setup_training_environment, estimate_training_time

logger = logging.getLogger(__name__)


def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Llama-3.2-Korean Fine-tuning")

    parser.add_argument("--csv_path", type=str, default=training_config.csv_path,
                        help="CSV 데이터 파일 경로")
    parser.add_argument("--output_dir", type=str, default=training_config.output_dir,
                        help="모델 저장 디렉토리")
    parser.add_argument("--model_name", type=str, default=model_config.model_name,
                        help="베이스 모델 이름")
    parser.add_argument("--batch_size", type=int, default=training_config.batch_size,
                        help="배치 크기")
    parser.add_argument("--epochs", type=int, default=training_config.num_train_epochs,
                        help="훈련 에폭 수")
    parser.add_argument("--learning_rate", type=float, default=training_config.learning_rate,
                        help="학습률")
    parser.add_argument("--max_length", type=int, default=model_config.max_length,
                        help="최대 시퀀스 길이")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="로그 레벨")
    parser.add_argument("--skip_validation", action="store_true",
                        help="시스템 요구사항 검사 건너뛰기")
    parser.add_argument("--dry_run", action="store_true",
                        help="실제 훈련 없이 설정만 확인")

    return parser.parse_args()


def update_configs(args):
    """설정 업데이트"""
    # 훈련 설정 업데이트
    training_config.csv_path = args.csv_path
    training_config.output_dir = args.output_dir
    training_config.batch_size = args.batch_size
    training_config.num_train_epochs = args.epochs
    training_config.learning_rate = args.learning_rate

    # 모델 설정 업데이트
    model_config.model_name = args.model_name
    model_config.max_length = args.max_length

    logger.info("Configuration updated with command line arguments")
    logger.info(f"Using model: {model_config.model_name}")


def validate_setup(args):
    """설정 검증"""
    logger.info("Validating setup...")

    # CSV 파일 검증
    if not validate_csv_file(args.csv_path):
        logger.error("CSV file validation failed")
        return False

    # 출력 디렉토리 생성
    create_directory(args.output_dir)

    # 시스템 요구사항 확인
    if not args.skip_validation:
        if not check_system_requirements():
            logger.error("System requirements check failed")
            return False

    logger.info("Setup validation completed")
    return True


def main():
    """메인 함수"""
    # 시작 시간 기록
    start_time = datetime.now()

    # 명령행 인자 파싱
    args = parse_arguments()

    # 로깅 설정
    log_file = f"training_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, log_file)

    logger.info("=" * 60)
    logger.info("🚀 Llama-3.2-Korean Fine-tuning Started")
    logger.info("=" * 60)

    try:
        # 환경 변수 설정
        set_environment_variables()

        # 시스템 정보 출력
        print_system_info()

        # 설정 업데이트
        update_configs(args)

        # 설정 검증
        if not validate_setup(args):
            logger.error("Setup validation failed")
            sys.exit(1)

        # GPU 메모리 정리
        clear_gpu_memory()

        # Dry run 모드
        if args.dry_run:
            logger.info("🧪 Dry run mode - configuration validation only")
            logger.info(f"CSV file: {args.csv_path} ✓")
            logger.info(f"Output directory: {args.output_dir} ✓")
            logger.info(f"Model: {args.model_name} ✓")
            logger.info(f"Batch size: {args.batch_size}")
            logger.info(f"Epochs: {args.epochs}")
            logger.info(f"Learning rate: {args.learning_rate}")
            logger.info("Configuration validation completed successfully!")
            return

        # 훈련 환경 설정
        setup_training_environment()

        # 1. 모델 매니저 생성 및 모델 로드
        logger.info("📦 Loading model and tokenizer...")
        model_manager = create_model_manager()
        model, tokenizer = model_manager.load_model_and_tokenizer()

        # 모델 크기 정보
        param_count = model.num_parameters()
        model_size = estimate_model_size(param_count, model_config.torch_dtype)
        logger.info(f"Model parameters: {param_count:,}")
        logger.info(f"Estimated model size: {model_size}")

        # 2. 데이터 로드 및 전처리
        logger.info("📊 Loading and preprocessing data...")
        tokenized_dataset = load_and_prepare_data(tokenizer, args.csv_path)

        # 훈련 시간 추정
        estimated_time = estimate_training_time(
            len(tokenized_dataset),
            args.batch_size * training_config.gradient_accumulation_steps,
            args.epochs
        )
        logger.info(f"⏱️ Estimated training time: {estimated_time}")

        # 3. 트레이너 생성
        logger.info("🏋️ Setting up trainer...")
        training_manager = create_training_manager(model, tokenizer)

        # 4. GPU 메모리 확인
        check_gpu_memory()

        # 5. 훈련 시작
        logger.info("🎯 Starting training...")
        train_result = training_manager.train(tokenized_dataset, args.output_dir)

        # 6. 모델 저장
        logger.info("💾 Saving final model...")
        training_manager.save_model(args.output_dir)

        # 훈련 완료 정보
        end_time = datetime.now()
        total_time = end_time - start_time

        logger.info("=" * 60)
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {args.output_dir}")
        logger.info(f"⏱️ Total training time: {total_time}")
        logger.info(f"📊 Final training loss: {train_result.training_loss:.4f}")
        logger.info(f"📈 Total training steps: {train_result.global_step}")
        logger.info("=" * 60)

        # 훈련 통계 저장
        training_stats = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time_seconds": total_time.total_seconds(),
            "final_loss": train_result.training_loss,
            "total_steps": train_result.global_step,
            "dataset_size": len(tokenized_dataset),
            "model_parameters": param_count,
            "config": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "model_name": args.model_name
            }
        }

        stats_file = os.path.join(args.output_dir, "training_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📈 Training statistics saved to: {stats_file}")

        # 빠른 테스트
        logger.info("🧪 Running quick inference test...")
        from inference_manager import create_inference_manager

        inference_manager = create_inference_manager(model, tokenizer)
        test_question = "전세 사기를 당했을 때 어떻게 대응해야 하나요?"
        test_response = inference_manager.generate_response(test_question)

        print("\n" + "=" * 50)
        print("🧪 Quick Test Result")
        print("=" * 50)
        print(f"질문: {test_question}")
        print(f"답변: {test_response}")
        print("=" * 50)

    except KeyboardInterrupt:
        logger.info("\n❌ Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    finally:
        # 메모리 정리
        clear_gpu_memory()

        # 최종 메모리 상태 확인
        if logger.isEnabledFor(logging.INFO):
            check_gpu_memory()


if __name__ == "__main__":
    main()