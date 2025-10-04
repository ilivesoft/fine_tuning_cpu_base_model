#!/usr/bin/env python3
"""
Windows CPU 전용 훈련 스크립트
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# GPU 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# CPU 전용 모듈 임포트
from config_cpu import training_config, model_config
from utils import (
    setup_logging,
    print_system_info,
    validate_csv_file,
    create_directory,
    ProgressTracker
)
from model_manager_cpu import create_cpu_model_manager, check_cpu_memory
from data_loader import load_and_prepare_data
from trainer import create_training_manager

logger = logging.getLogger(__name__)

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="CPU 전용 Llama Fine-tuning")
    
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
    parser.add_argument("--max_steps", type=int, default=training_config.max_steps,
                       help="최대 스텝 수 (CPU 최적화)")
    parser.add_argument("--learning_rate", type=float, default=training_config.learning_rate,
                       help="학습률")
    parser.add_argument("--save_steps", type=int, default=training_config.save_steps,
                       help="저장 간격")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="로그 레벨")
    parser.add_argument("--dry_run", action="store_true",
                       help="설정 검증만 실행")
    
    return parser.parse_args()

def update_configs(args):
    """설정 업데이트"""
    training_config.csv_path = args.csv_path
    training_config.output_dir = args.output_dir
    training_config.batch_size = args.batch_size
    training_config.num_train_epochs = args.epochs
    training_config.max_steps = args.max_steps
    training_config.learning_rate = args.learning_rate
    training_config.save_steps = args.save_steps
    
    model_config.model_name = args.model_name
    
    logger.info("Configuration updated")
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Max steps: {training_config.max_steps}")

def validate_setup(args):
    """설정 검증"""
    logger.info("Validating setup...")
    
    # CSV 파일 검증
    if not validate_csv_file(args.csv_path):
        logger.error("CSV file validation failed")
        return False
    
    # 출력 디렉토리 생성
    create_directory(args.output_dir)
    
    logger.info("Setup validation completed")
    return True

def main():
    """메인 함수"""
    start_time = datetime.now()
    
    # 명령행 인자 파싱
    args = parse_arguments()
    
    # 로깅 설정
    log_file = f"training_cpu_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, log_file)
    
    logger.info("=" * 60)
    logger.info("🖥️ CPU 전용 Llama Fine-tuning Started")
    logger.info("=" * 60)
    
    try:
        # 시스템 정보 출력
        print_system_info()
        
        # 설정 업데이트
        update_configs(args)
        
        # 설정 검증
        if not validate_setup(args):
            logger.error("Setup validation failed")
            sys.exit(1)
        
        # Dry run 모드
        if args.dry_run:
            logger.info("🧪 Dry run mode - configuration validation only")
            logger.info(f"CSV file: {args.csv_path} ✓")
            logger.info(f"Output directory: {args.output_dir} ✓")
            logger.info(f"Model: {args.model_name} ✓")
            logger.info(f"Max steps: {args.max_steps}")
            logger.info("Configuration validation completed!")
            return
        
        # CPU 메모리 확인
        check_cpu_memory()
        
        # 1. 모델 매니저 생성 및 모델 로드
        logger.info("📦 Loading model and tokenizer on CPU...")
        model_manager = create_cpu_model_manager()
        model, tokenizer = model_manager.load_model_and_tokenizer()
        
        # 2. 데이터 로드
        logger.info("📊 Loading and preprocessing data...")
        tokenized_dataset = load_and_prepare_data(tokenizer, args.csv_path)
        
        # 3. 트레이너 생성
        logger.info("🏋️ Setting up trainer...")
        training_manager = create_training_manager(model, tokenizer)
        
        # 4. 훈련 시작
        logger.info("🎯 Starting training on CPU...")
        logger.warning("⚠️ CPU training is slow. Please be patient...")
        
        train_result = training_manager.train(tokenized_dataset, args.output_dir)
        
        # 5. 모델 저장
        logger.info("💾 Saving final model...")
        training_manager.save_model(args.output_dir)
        
        # 훈련 완료 정보
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {args.output_dir}")
        logger.info(f"⏱️ Total time: {total_time}")
        logger.info(f"📊 Final loss: {train_result.training_loss:.4f}")
        logger.info(f"📈 Total steps: {train_result.global_step}")
        logger.info("=" * 60)
        
        # 훈련 통계 저장
        training_stats = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time_seconds": total_time.total_seconds(),
            "final_loss": train_result.training_loss,
            "total_steps": train_result.global_step,
            "dataset_size": len(tokenized_dataset),
            "device": "cpu",
            "config": {
                "batch_size": args.batch_size,
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "model_name": args.model_name
            }
        }
        
        stats_file = os.path.join(args.output_dir, "training_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📈 Training statistics saved to: {stats_file}")
        
    except KeyboardInterrupt:
        logger.info("\n❌ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
