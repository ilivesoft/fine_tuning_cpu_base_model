"""
Utility functions for the project
"""

import os
import json
import logging
import subprocess
import torch
from typing import Dict, List, Optional, Any
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """로깅 설정"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 로그 레벨 설정
    level = getattr(logging, log_level.upper())
    
    # 핸들러 설정
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    # 기본 로깅 설정
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # transformers 라이브러리 로그 레벨 조정
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("tokenizers").setLevel(logging.WARNING)
    
    logger.info(f"Logging setup completed with level: {log_level}")

def check_system_requirements():
    """시스템 요구사항 확인"""
    logger.info("Checking system requirements...")
    
    # GPU 확인
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be very slow on CPU.")
        return False
    
    # GPU 메모리 확인
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 6:
        logger.warning("GPU memory is less than 6GB. Training may fail.")
        return False
    
    # RAM 확인
    ram_gb = psutil.virtual_memory().total / 1024**3
    logger.info(f"System RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 12:
        logger.warning("System RAM is less than 12GB. Consider closing other applications.")
    
    # 디스크 공간 확인
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / 1024**3
    logger.info(f"Free disk space: {free_gb:.1f} GB")
    
    if free_gb < 10:
        logger.warning("Free disk space is less than 10GB. Model saving may fail.")
        return False
    
    logger.info("System requirements check passed!")
    return True

def get_gpu_info() -> Dict[str, Any]:
    """GPU 정보 반환"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    gpu_info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        memory_total = device_props.total_memory / 1024**3
        
        device_info = {
            "id": i,
            "name": device_props.name,
            "total_memory_gb": memory_total,
            "allocated_memory_gb": memory_allocated,
            "reserved_memory_gb": memory_reserved,
            "free_memory_gb": memory_total - memory_reserved,
            "compute_capability": f"{device_props.major}.{device_props.minor}"
        }
        
        gpu_info["devices"].append(device_info)
    
    return gpu_info

def print_gpu_info():
    """GPU 정보 출력"""
    gpu_info = get_gpu_info()
    
    if not gpu_info["available"]:
        print("❌ CUDA not available")
        return
    
    print(f"\n🎮 GPU Information:")
    print(f"   Device count: {gpu_info['device_count']}")
    
    for device in gpu_info["devices"]:
        print(f"\n   GPU {device['id']}: {device['name']}")
        print(f"   - Total Memory: {device['total_memory_gb']:.1f} GB")
        print(f"   - Free Memory: {device['free_memory_gb']:.1f} GB")
        print(f"   - Compute Capability: {device['compute_capability']}")

def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")

def set_environment_variables():
    """환경 변수 설정"""
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "warning"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")

def create_directory(path: str):
    """디렉토리 생성"""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Directory created/verified: {path}")

def save_config_to_file(config_dict: Dict, filepath: str):
    """설정을 JSON 파일로 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Config saved to {filepath}")

def load_config_from_file(filepath: str) -> Dict:
    """JSON 파일에서 설정 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logger.info(f"Config loaded from {filepath}")
    return config

def estimate_model_size(num_parameters: int, dtype: torch.dtype = torch.float16) -> str:
    """모델 크기 추정"""
    if dtype == torch.float16:
        bytes_per_param = 2
    elif dtype == torch.float32:
        bytes_per_param = 4
    elif dtype == torch.int8:
        bytes_per_param = 1
    else:
        bytes_per_param = 4  # 기본값
    
    total_bytes = num_parameters * bytes_per_param
    
    if total_bytes < 1024**3:
        return f"{total_bytes / 1024**2:.1f} MB"
    else:
        return f"{total_bytes / 1024**3:.1f} GB"

def format_time_duration(seconds: float) -> str:
    """시간 지속시간 포맷팅"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}시간 {minutes}분 {secs}초"
    elif minutes > 0:
        return f"{minutes}분 {secs}초"
    else:
        return f"{secs}초"

def get_system_info() -> Dict[str, Any]:
    """시스템 정보 수집"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": subprocess.check_output(["python3", "--version"]).decode().strip(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024**3,
            "disk_free_gb": psutil.disk_usage('.').free / 1024**3
        }
    }
    
    if torch.cuda.is_available():
        info["gpu"] = get_gpu_info()
    
    return info

def print_system_info():
    """시스템 정보 출력"""
    info = get_system_info()
    
    print("\n" + "="*50)
    print("💻 System Information")
    print("="*50)
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['torch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"CPU Cores: {info['system']['cpu_count']}")
    print(f"RAM: {info['system']['memory_gb']:.1f} GB")
    print(f"Free Disk: {info['system']['disk_free_gb']:.1f} GB")
    
    if info['cuda_available']:
        print_gpu_info()
    
    print("="*50)

def validate_csv_file(csv_path: str) -> bool:
    """CSV 파일 유효성 검사"""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        required_columns = ['question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if len(df) == 0:
            logger.error("CSV file is empty")
            return False
        
        logger.info(f"CSV file validation passed: {len(df)} rows")
        return True
        
    except Exception as e:
        logger.error(f"Error validating CSV file: {e}")
        return False

def backup_model(model_path: str, backup_dir: str = "./backups"):
    """모델 백업"""
    if not os.path.exists(model_path):
        logger.warning(f"Model path not found: {model_path}")
        return
    
    create_directory(backup_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"model_backup_{timestamp}")
    
    import shutil
    shutil.copytree(model_path, backup_path)
    logger.info(f"Model backed up to: {backup_path}")

def cleanup_checkpoints(output_dir: str, keep_latest: int = 2):
    """체크포인트 정리"""
    if not os.path.exists(output_dir):
        return
    
    checkpoint_dirs = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    
    if len(checkpoint_dirs) <= keep_latest:
        return
    
    # 체크포인트 번호로 정렬
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
    
    # 오래된 체크포인트 삭제
    for checkpoint_dir in checkpoint_dirs[:-keep_latest]:
        checkpoint_path = os.path.join(output_dir, checkpoint_dir)
        import shutil
        shutil.rmtree(checkpoint_path)
        logger.info(f"Removed old checkpoint: {checkpoint_path}")

class ProgressTracker:
    """진행 상황 추적 클래스"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        
    def update(self, step: int):
        """진행 상황 업데이트"""
        self.current_step = step
        
    def get_progress_info(self) -> Dict[str, Any]:
        """진행 상황 정보 반환"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        progress_ratio = self.current_step / self.total_steps if self.total_steps > 0 else 0
        
        if progress_ratio > 0:
            estimated_total_time = elapsed_time / progress_ratio
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": progress_ratio * 100,
            "elapsed_time": format_time_duration(elapsed_time),
            "remaining_time": format_time_duration(remaining_time) if remaining_time > 0 else "계산 중...",
        }
    
    def print_progress(self):
        """진행 상황 출력"""
        info = self.get_progress_info()
        print(f"진행률: {info['progress_percent']:.1f}% ({info['current_step']}/{info['total_steps']})")
        print(f"경과 시간: {info['elapsed_time']}")
        print(f"예상 남은 시간: {info['remaining_time']}")
