#!/bin/bash
# 실행 스크립트 모음

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# 함수: 헤더 출력
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# 함수: 성공 메시지
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 함수: 에러 메시지
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 함수: 경고 메시지
print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# 함수: 정보 메시지
print_info() {
    echo -e "${CYAN}ℹ️ $1${NC}"
}

# GPU 메모리 정리
clear_gpu() {
    print_info "GPU 메모리 정리 중..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# 시스템 정보 확인
check_system() {
    print_header "시스템 정보 확인"
    
    echo -e "${PURPLE}Python 버전:${NC}"
    python3 --version
    
    echo -e "${PURPLE}GPU 정보:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    else
        print_warning "nvidia-smi not found"
    fi
    
    echo -e "${PURPLE}CUDA 사용 가능:${NC}"
    python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || print_error "PyTorch not installed"
}

# CSV 파일 확인
check_csv() {
    local csv_file=${1:-"civil_law_qa_dataset.csv"}
    
    if [ -f "$csv_file" ]; then
        print_success "CSV 파일 발견: $csv_file"
        
        # 파일 정보 출력
        local line_count=$(wc -l < "$csv_file")
        echo -e "${CYAN}총 라인 수: $line_count${NC}"
        
        # 헤더 확인
        echo -e "${CYAN}헤더:${NC}"
        head -n 1 "$csv_file"
        
    else
        print_error "CSV 파일이 없습니다: $csv_file"
        print_info "먼저 CSV 파일을 생성해주세요."
        return 1
    fi
}

# 훈련 실행
run_training() {
    local csv_path=${1:-"civil_law_qa_dataset.csv"}
    local output_dir=${2:-"./fine_tuned_model"}
    
    print_header "Fine-tuning 훈련 시작"
    
    # 전처리
    clear_gpu
    check_csv "$csv_path" || return 1
    
    print_info "훈련 시작..."
    python3 main_train.py \
        --csv_path "$csv_path" \
        --output_dir "$output_dir" \
        --log_level INFO
    
    if [ $? -eq 0 ]; then
        print_success "훈련 완료!"
        print_info "모델 저장 위치: $output_dir"
    else
        print_error "훈련 실패!"
        return 1
    fi
}

# 훈련 (dry run)
run_training_dry() {
    local csv_path=${1:-"civil_law_qa_dataset.csv"}
    
    print_header "설정 검증 (Dry Run)"
    
    python3 main_train.py \
        --csv_path "$csv_path" \
        --dry_run \
        --log_level INFO
    
    if [ $? -eq 0 ]; then
        print_success "설정 검증 완료!"
    else
        print_error "설정 검증 실패!"
        return 1
    fi
}

# 추론 (대화형)
run_inference_interactive() {
    local model_path=${1:-"./fine_tuned_model"}
    
    print_header "대화형 추론 시작"
    
    if [ ! -d "$model_path" ]; then
        print_error "모델 디렉토리가 없습니다: $model_path"
        return 1
    fi
    
    clear_gpu
    python3 main_inference.py \
        --model_path "$model_path" \
        --interactive \
        --log_level INFO
}

# 추론 (단일 질문)
run_inference_single() {
    local model_path=${1:-"./fine_tuned_model"}
    local question=${2:-"전세 사기를 당했을 때 어떻게 대응해야 하나요?"}
    
    print_header "단일 질문 추론"
    
    if [ ! -d "$model_path" ]; then
        print_error "모델 디렉토리가 없습니다: $model_path"
        return 1
    fi
    
    clear_gpu
    python3 main_inference.py \
        --model_path "$model_path" \
        --question "$question" \
        --log_level INFO
}

# 추론 (벤치마크)
run_inference_benchmark() {
    local model_path=${1:-"./fine_tuned_model"}
    
    print_header "벤치마크 테스트 시작"
    
    if [ ! -d "$model_path" ]; then
        print_error "모델 디렉토리가 없습니다: $model_path"
        return 1
    fi
    
    clear_gpu
    python3 main_inference.py \
        --model_path "$model_path" \
        --benchmark \
        --log_level INFO
}

# 빠른 테스트
run_quick_test() {
    local model_path=${1:-"./fine_tuned_model"}
    
    print_header "빠른 테스트 시작"
    
    if [ ! -d "$model_path" ]; then
        print_error "모델 디렉토리가 없습니다: $model_path"
        return 1
    fi
    
    clear_gpu
    python3 main_inference.py \
        --model_path "$model_path" \
        --quick_test \
        --log_level WARNING
}

# 전체 파이프라인 실행
run_full_pipeline() {
    local csv_path=${1:-"civil_law_qa_dataset.csv"}
    local output_dir=${2:-"./fine_tuned_model"}
    
    print_header "전체 파이프라인 실행"
    
    # 1. 시스템 확인
    check_system
    
    # 2. 설정 검증
    print_info "1/3: 설정 검증 중..."
    run_training_dry "$csv_path" || return 1
    
    # 3. 훈련 실행
    print_info "2/3: 훈련 시작..."
    run_training "$csv_path" "$output_dir" || return 1
    
    # 4. 빠른 테스트
    print_info "3/3: 빠른 테스트..."
    run_quick_test "$output_dir" || return 1
    
    print_success "전체 파이프라인 완료!"
}

# 도움말
show_help() {
    echo -e "${BLUE}📖 사용법:${NC}"
    echo "  $0 [명령] [인자...]"
    echo ""
    echo -e "${YELLOW}명령어:${NC}"
    echo "  check_system              - 시스템 정보 확인"
    echo "  check_csv [파일경로]      - CSV 파일 확인"
    echo "  train [CSV] [출력디렉토리] - 훈련 실행"
    echo "  train_dry [CSV]           - 설정 검증만 실행"
    echo "  chat [모델경로]           - 대화형 추론"
    echo "  ask [모델경로] [질문]     - 단일 질문 추론"
    echo "  benchmark [모델경로]      - 벤치마크 테스트"
    echo "  test [모델경로]           - 빠른 테스트"
    echo "  full [CSV] [출력디렉토리] - 전체 파이프라인"
    echo "  help                      - 이 도움말"
    echo ""
    echo -e "${YELLOW}예시:${NC}"
    echo "  $0 check_system"
    echo "  $0 train"
    echo "  $0 chat"
    echo "  $0 ask ./fine_tuned_model \"전세권이란 무엇인가요?\""
    echo "  $0 full"
}

# 메인 함수
main() {
    case ${1:-help} in
        check_system|system)
            check_system
            ;;
        check_csv|csv)
            check_csv "$2"
            ;;
        train)
            run_training "$2" "$3"
            ;;
        train_dry|dry)
            run_training_dry "$2"
            ;;
        chat|interactive)
            run_inference_interactive "$2"
            ;;
        ask|question)
            run_inference_single "$2" "$3"
            ;;
        benchmark|bench)
            run_inference_benchmark "$2"
            ;;
        test|quick)
            run_quick_test "$2"
            ;;
        full|pipeline)
            run_full_pipeline "$2" "$3"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "알 수 없는 명령: $1"
            show_help
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@"
