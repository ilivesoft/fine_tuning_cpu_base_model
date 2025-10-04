"""
Data loading and preprocessing utilities
"""

import pandas as pd
import logging
from datasets import Dataset
from typing import Dict, List
from transformers import PreTrainedTokenizer

from config import format_prompt, training_config

logger = logging.getLogger(__name__)

class DataLoader:
    """데이터 로딩 및 전처리 클래스"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_csv_data(self, csv_path: str) -> Dataset:
        """CSV 파일에서 데이터 로드"""
        logger.info(f"Loading data from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # 필요한 컬럼 확인
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # 결측값 제거
            initial_length = len(df)
            df = df.dropna(subset=['question', 'answer'])
            if len(df) < initial_length:
                logger.warning(f"Removed {initial_length - len(df)} rows with missing values")
            
            # 빈 문자열 제거
            df = df[(df['question'].str.strip() != '') & (df['answer'].str.strip() != '')]
            logger.info(f"Final dataset size: {len(df)} rows")
            
            return self._create_dataset(df)
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise
    
    def _create_dataset(self, df: pd.DataFrame) -> Dataset:
        """DataFrame을 Dataset으로 변환"""
        prompts = []
        
        for _, row in df.iterrows():
            prompt = format_prompt(
                question=row['question'].strip(),
                answer=row['answer'].strip()
            )
            prompts.append(prompt)
        
        dataset = Dataset.from_dict({"text": prompts})
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """데이터셋 토크나이징"""
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            # labels 설정 (causal LM이므로 input_ids와 동일)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def get_data_statistics(self, dataset: Dataset) -> Dict:
        """데이터 통계 정보 반환"""
        text_lengths = [len(text.split()) for text in dataset['text']]
        
        stats = {
            'total_samples': len(dataset),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths),
        }
        
        return stats
    
    def print_sample_data(self, dataset: Dataset, num_samples: int = 3):
        """샘플 데이터 출력"""
        logger.info(f"Sample data (first {num_samples} samples):")
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]['text']
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(sample[:500] + "..." if len(sample) > 500 else sample)

def create_data_loader(tokenizer: PreTrainedTokenizer, max_length: int = None) -> DataLoader:
    """DataLoader 인스턴스 생성"""
    if max_length is None:
        max_length = training_config.batch_size
    
    return DataLoader(tokenizer, max_length)

def load_and_prepare_data(tokenizer: PreTrainedTokenizer, csv_path: str = None) -> Dataset:
    """데이터 로드 및 전처리 원스톱 함수"""
    if csv_path is None:
        csv_path = training_config.csv_path
    
    data_loader = create_data_loader(tokenizer)
    
    # 데이터 로드
    dataset = data_loader.load_csv_data(csv_path)
    
    # 통계 정보 출력
    stats = data_loader.get_data_statistics(dataset)
    logger.info(f"Dataset statistics: {stats}")
    
    # 샘플 데이터 출력
    data_loader.print_sample_data(dataset)
    
    # 토크나이징
    tokenized_dataset = data_loader.tokenize_dataset(dataset)
    
    return tokenized_dataset
