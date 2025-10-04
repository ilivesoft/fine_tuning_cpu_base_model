#!/usr/bin/env python3
"""
Windows CPU 전용 추론 스크립트
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# GPU 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from config_cpu import model_config, inference_config
from utils import (
    setup_logging,
    print_system_info,
    create_directory
)
from model_manager_cpu import create_inference_manager, check_cpu_memory
from inference_manager import create_inference_manager as create_inf_mgr, create_chatbot

logger = logging.getLogger(__name__)

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="CPU 전용 추론")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Fine-tuned 모델 경로")
    parser.add_argument("--base_model", type=str, default=model_config.model_name,
                       help="베이스 모델 이름")
    parser.add_argument("--question", type=str,
                       help="단일 질문")
    parser.add_argument("--interactive", action="store_true",
                       help="대화형 모드")
    parser.add_argument("--quick_test", action="store_true",
                       help="빠른 테스트")
    parser.add_argument("--max_new_tokens", type=int, default=inference_config.max_new_tokens,
                       help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=inference_config.temperature,
                       help="생성 온도")
    parser.add_argument("--output_file", type=str,
                       help="결과 저장 파일")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="로그 레벨")
    
    return parser.parse_args()

def validate_arguments(args):
    """인자 유효성 검사"""
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return False
    
    modes = [args.question, args.interactive, args.quick_test]
    if sum(bool(mode) for mode in modes) == 0:
        logger.warning("No operation mode specified. Running quick test.")
        args.quick_test = True
    
    return True

def run_single_question(inference_manager, question: str, args):
    """단일 질문 처리"""
    logger.info(f"Processing question: {question[:50]}...")
    
    response = inference_manager.generate_response(
        question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print("\n" + "="*60)
    print("📋 Result")
    print("="*60)
    print(f"질문: {question}")
    print(f"\n답변: {response}")
    print("="*60)
    
    return [{"question": question, "response": response}]

def run_quick_test(inference_manager):
    """빠른 테스트"""
    test_questions = [
        "전세권이란 무엇인가요?",
        "전세 사기를 예방하는 방법은?"
    ]
    
    print("\n" + "="*60)
    print("🧪 Quick Test")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. 질문: {question}")
        response = inference_manager.generate_response(question)
        print(f"   답변: {response}")
        print("-" * 60)

def main():
    """메인 함수"""
    start_time = datetime.now()
    
    args = parse_arguments()
    
    # 로깅 설정
    log_file = f"inference_cpu_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, log_file)
    
    logger.info("="*60)
    logger.info("🖥️ CPU Inference Started")
    logger.info("="*60)
    
    try:
        # 인자 검증
        if not validate_arguments(args):
            logger.error("Argument validation failed")
            sys.exit(1)
        
        # 시스템 정보
        print_system_info()
        
        # CPU 메모리 확인
        check_cpu_memory()
        
        # 모델 로드
        logger.info(f"📦 Loading model from: {args.model_path}")
        model_loader = create_inference_manager(args.model_path, args.base_model)
        model, tokenizer = model_loader.load_finetuned_model()
        
        # 추론 매니저 생성
        inference_manager = create_inf_mgr(model, tokenizer)
        
        # 생성 설정 업데이트
        inference_manager.update_generation_config(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        results = []
        
        # 실행 모드
        if args.interactive:
            logger.info("🗣️ Starting interactive mode...")
            chatbot = create_chatbot(inference_manager)
            conversation_history = inference_manager.interactive_chat()
            
            if args.output_file:
                chatbot.save_conversation(args.output_file)
        
        elif args.quick_test:
            logger.info("🧪 Running quick test...")
            run_quick_test(inference_manager)
        
        elif args.question:
            results = run_single_question(inference_manager, args.question, args)
        
        # 완료 정보
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info("="*60)
        logger.info("🎉 Inference completed!")
        logger.info(f"⏱️ Total time: {total_time}")
        if args.output_file:
            logger.info(f"📁 Results saved to: {args.output_file}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\n❌ Interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
