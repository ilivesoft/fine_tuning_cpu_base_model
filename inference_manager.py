"""
Inference and text generation utilities
"""

import torch
import logging
from typing import Dict, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from config_cpu import inference_config, format_prompt

logger = logging.getLogger(__name__)

class InferenceManager:
    """추론 관리 클래스"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # 추론 설정
        self.generation_config = {
            "max_new_tokens": inference_config.max_new_tokens,
            "temperature": inference_config.temperature,
            "do_sample": inference_config.do_sample,
            "top_p": inference_config.top_p,
            "repetition_penalty": inference_config.repetition_penalty,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
    def generate_response(self, question: str, **generation_kwargs) -> str:
        """질문에 대한 답변 생성"""
        # 프롬프트 생성
        prompt = format_prompt(question)
        
        # 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 생성 설정 업데이트
        gen_config = self.generation_config.copy()
        gen_config.update(generation_kwargs)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config
                )
            
            # 새로 생성된 토큰만 디코딩
            new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # 후처리
            response = self._post_process_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    def _post_process_response(self, response: str) -> str:
        """응답 후처리"""
        # 불필요한 공백 제거
        response = response.strip()
        
        # 특수 토큰 제거 (혹시 남아있는 경우)
        special_tokens = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
        for token in special_tokens:
            response = response.replace(token, "")
        
        return response.strip()
    
    def generate_batch_responses(self, questions: List[str], **generation_kwargs) -> List[str]:
        """여러 질문에 대한 배치 응답 생성"""
        responses = []
        
        for question in questions:
            try:
                response = self.generate_response(question, **generation_kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error generating response for question: {question[:50]}...")
                responses.append("답변 생성 중 오류가 발생했습니다.")
        
        return responses
    
    def interactive_chat(self):
        """대화형 채팅 인터페이스"""
        print("\n" + "="*60)
        print("🏠 민법 부동산 법률 상담 챗봇")
        print("="*60)
        print("질문을 입력하세요. 종료하려면 'quit', 'exit', '종료', '나가기'를 입력하세요.")
        print("-"*60)
        
        conversation_history = []
        
        while True:
            try:
                # 사용자 입력
                question = input("\n💬 질문: ").strip()
                
                # 종료 명령 확인
                if question.lower() in ['quit', 'exit', '종료', '나가기']:
                    print("\n👋 상담을 종료합니다. 감사합니다!")
                    break
                
                if not question:
                    print("질문을 입력해주세요.")
                    continue
                
                # 답변 생성
                print("🤔 답변 생성 중...")
                response = self.generate_response(question)
                
                # 결과 출력
                print(f"\n🏛️ 답변:\n{response}")
                print("\n" + "-"*60)
                
                # 대화 기록 저장
                conversation_history.append({
                    "question": question,
                    "response": response
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 상담을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류가 발생했습니다: {e}")
                logger.error(f"Interactive chat error: {e}")
        
        return conversation_history
    
    def benchmark_generation(self, test_questions: List[str] = None) -> Dict:
        """생성 성능 벤치마크"""
        if test_questions is None:
            test_questions = [
                "전세 사기를 당했을 때 어떻게 대응해야 하나요?",
                "전세권 등기의 효력은 무엇인가요?",
                "깡통전세의 위험성과 예방 방법을 알려주세요."
            ]
        
        logger.info("Running generation benchmark...")
        
        results = {
            "questions": [],
            "responses": [],
            "generation_times": [],
            "token_counts": []
        }
        
        for question in test_questions:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            # 시간 측정 시작
            start_time.record()
            
            # 응답 생성
            response = self.generate_response(question)
            
            # 시간 측정 종료
            end_time.record()
            torch.cuda.synchronize()
            
            generation_time = start_time.elapsed_time(end_time) / 1000.0  # 초 단위
            token_count = len(self.tokenizer.encode(response))
            
            results["questions"].append(question)
            results["responses"].append(response)
            results["generation_times"].append(generation_time)
            results["token_counts"].append(token_count)
            
            logger.info(f"Generated {token_count} tokens in {generation_time:.2f}s")
        
        # 평균 통계
        avg_time = sum(results["generation_times"]) / len(results["generation_times"])
        avg_tokens = sum(results["token_counts"]) / len(results["token_counts"])
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        results["average_generation_time"] = avg_time
        results["average_token_count"] = avg_tokens
        results["tokens_per_second"] = tokens_per_second
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Average generation time: {avg_time:.2f}s")
        logger.info(f"  Average tokens: {avg_tokens:.1f}")
        logger.info(f"  Tokens per second: {tokens_per_second:.1f}")
        
        return results
    
    def update_generation_config(self, **kwargs):
        """생성 설정 업데이트"""
        self.generation_config.update(kwargs)
        logger.info(f"Updated generation config: {kwargs}")

class ChatBot:
    """법률 상담 챗봇 클래스"""
    
    def __init__(self, inference_manager: InferenceManager):
        self.inference_manager = inference_manager
        self.conversation_history = []
        
    def chat(self, question: str) -> str:
        """단일 질문 처리"""
        response = self.inference_manager.generate_response(question)
        
        # 대화 기록 저장
        self.conversation_history.append({
            "question": question,
            "response": response,
            "timestamp": torch.cuda.Event(enable_timing=True)
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 기록 반환"""
        return self.conversation_history
    
    def clear_history(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def save_conversation(self, filepath: str):
        """대화 기록 저장"""
        import json
        
        # 타임스탬프는 저장에서 제외
        history_to_save = [
            {"question": item["question"], "response": item["response"]}
            for item in self.conversation_history
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Conversation saved to {filepath}")

def create_inference_manager(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> InferenceManager:
    """InferenceManager 인스턴스 생성"""
    return InferenceManager(model, tokenizer)

def create_chatbot(inference_manager: InferenceManager) -> ChatBot:
    """ChatBot 인스턴스 생성"""
    return ChatBot(inference_manager)

def run_quick_test(inference_manager: InferenceManager):
    """빠른 테스트 실행"""
    test_questions = [
        "전세 계약 시 주의사항은 무엇인가요?",
        "전세권과 임차권의 차이점을 설명해주세요."
    ]
    
    print("\n" + "="*50)
    print("🧪 Quick Test")
    print("="*50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. 질문: {question}")
        response = inference_manager.generate_response(question)
        print(f"   답변: {response}")
        print("-" * 50)