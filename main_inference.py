#!/usr/bin/env python3
"""
메인 추론 스크립트
Fine-tuned Llama 모델을 사용한 법률 상담
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 프로젝트 모듈 임포트
from config import model_config, inference_config
from utils import (
    setup_logging,
    print_system_info,
    set_environment_variables,
    clear_gpu_memory,
    create_directory
)
from model_manager import create_inference_manager, check_gpu_memory
from inference_manager import create_inference_manager as create_inf_mgr, create_chatbot, run_quick_test

logger = logging.getLogger(__name__)

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Fine-tuned Llama 모델 추론")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Fine-tuned 모델 경로")
    parser.add_argument("--base_model", type=str, default=model_config.model_name,
                       help="베이스 모델 이름")
    parser.add_argument("--question", type=str,
                       help="단일 질문")
    parser.add_argument("--questions_file", type=str,
                       help="질문들이 담긴 텍스트 파일")
    parser.add_argument("--interactive", action="store_true",
                       help="대화형 모드")
    parser.add_argument("--benchmark", action="store_true",
                       help="벤치마크 테스트 실행")
    parser.add_argument("--quick_test", action="store_true",
                       help="빠른 테스트 실행")
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
    # 모델 경로 확인
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return False
    
    # 질문 파일 확인
    if args.questions_file and not os.path.exists(args.questions_file):
        logger.error(f"Questions file does not exist: {args.questions_file}")
        return False
    
    # 모드 확인
    modes = [args.question, args.questions_file, args.interactive, args.benchmark, args.quick_test]
    if sum(bool(mode) for mode in modes) == 0:
        logger.warning("No operation mode specified. Running quick test.")
        args.quick_test = True
    
    return True

def load_questions_from_file(filepath: str) -> list:
    """파일에서 질문 목록 로드"""
    questions = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 빈 줄과 주석 제외
                    questions.append(line)
        logger.info(f"Loaded {len(questions)} questions from {filepath}")
    except Exception as e:
        logger.error(f"Error loading questions from file: {e}")
    return questions

def save_results_to_file(results: list, filepath: str):
    """결과를 파일에 저장"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 법률 상담 결과\n")
            f.write(f"# 생성 시간: {datetime.now().isoformat()}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"## 질문 {i}\n")
                f.write(f"**질문:** {result['question']}\n\n")
                f.write(f"**답변:** {result['response']}\n\n")
                f.write("-" * 80 + "\n\n")
        
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def run_single_question(inference_manager, question: str, args):
    """단일 질문 처리"""
    logger.info(f"Processing single question: {question[:50]}...")
    
    response = inference_manager.generate_response(
        question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("📋 Single Question Result")
    print("="*60)
    print(f"질문: {question}")
    print(f"\n답변: {response}")
    print("="*60)
    
    return [{"question": question, "response": response}]

def run_multiple_questions(inference_manager, questions: list, args):
    """여러 질문 처리"""
    logger.info(f"Processing {len(questions)} questions...")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n진행률: {i}/{len(questions)}")
        print(f"질문: {question}")
        
        response = inference_manager.generate_response(
            question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        print(f"답변: {response}")
        print("-" * 60)
        
        results.append({
            "question": question,
            "response": response
        })
    
    return results

def run_benchmark_test(inference_manager):
    """벤치마크 테스트 실행"""
    logger.info("Running benchmark test...")
    
    test_questions = [
        "전세 사기를 당했을 때 어떻게 대응해야 하나요?",
        "전세권 등기의 효력은 무엇인가요?",
        "깡통전세의 위험성과 예방 방법을 알려주세요.",
        "전세보증금 반환보증보험이란 무엇인가요?",
        "임대인이 파산한 경우 전세보증금을 어떻게 회수하나요?"
    ]
    
    benchmark_results = inference_manager.benchmark_generation(test_questions)
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 Benchmark Results")
    print("="*60)
    print(f"평균 생성 시간: {benchmark_results['average_generation_time']:.2f}초")
    print(f"평균 토큰 수: {benchmark_results['average_token_count']:.1f}")
    print(f"초당 토큰 수: {benchmark_results['tokens_per_second']:.1f}")
    print("\n세부 결과:")
    
    for i, (question, response, time, tokens) in enumerate(zip(
        benchmark_results['questions'],
        benchmark_results['responses'],
        benchmark_results['generation_times'],
        benchmark_results['token_counts']
    ), 1):
        print(f"\n{i}. 질문: {question}")
        print(f"   답변: {response}")
        print(f"   시간: {time:.2f}초, 토큰: {tokens}개")
    
    print("="*60)
    
    return benchmark_results

def main():
    """메인 함수"""
    # 시작 시간 기록
    start_time = datetime.now()
    
    # 명령행 인자 파싱
    args = parse_arguments()
    
    # 로깅 설정
    log_file = f"inference_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, log_file)
    
    logger.info("="*60)
    logger.info("🔮 Llama Inference Started")
    logger.info("="*60)
    
    try:
        # 환경 변수 설정
        set_environment_variables()
        
        # 인자 유효성 검사
        if not validate_arguments(args):
            logger.error("Argument validation failed")
            sys.exit(1)
        
        # 시스템 정보 출력
        print_system_info()
        
        # GPU 메모리 정리
        clear_gpu_memory()
        
        # 모델 로더 생성 및 모델 로드
        logger.info(f"📦 Loading fine-tuned model from: {args.model_path}")
        model_loader = create_inference_manager(args.model_path, args.base_model)
        model, tokenizer = model_loader.load_finetuned_model()
        
        # 추론 매니저 생성
        inference_manager = create_inf_mgr(model, tokenizer)
        
        # 생성 설정 업데이트
        inference_manager.update_generation_config(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # GPU 메모리 확인
        check_gpu_memory()
        
        results = []
        
        # 실행 모드에 따른 처리
        if args.interactive:
            # 대화형 모드
            logger.info("🗣️ Starting interactive mode...")
            chatbot = create_chatbot(inference_manager)
            conversation_history = inference_manager.interactive_chat()
            
            # 대화 기록 저장
            if args.output_file:
                chatbot.save_conversation(args.output_file)
        
        elif args.benchmark:
            # 벤치마크 모드
            benchmark_results = run_benchmark_test(inference_manager)
            results = [{"benchmark": benchmark_results}]
        
        elif args.quick_test:
            # 빠른 테스트 모드
            logger.info("🧪 Running quick test...")
            run_quick_test(inference_manager)
        
        elif args.question:
            # 단일 질문 모드
            results = run_single_question(inference_manager, args.question, args)
        
        elif args.questions_file:
            # 다중 질문 모드
            questions = load_questions_from_file(args.questions_file)
            if questions:
                results = run_multiple_questions(inference_manager, questions, args)
        
        # 결과 저장
        if results and args.output_file and not args.interactive:
            save_results_to_file(results, args.output_file)
        
        # 완료 정보
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info("="*60)
        logger.info("🎉 Inference completed successfully!")
        logger.info(f"⏱️ Total time: {total_time}")
        if args.output_file:
            logger.info(f"📁 Results saved to: {args.output_file}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\n❌ Inference interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Inference failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # 메모리 정리
        clear_gpu_memory()

if __name__ == "__main__":
    main()
