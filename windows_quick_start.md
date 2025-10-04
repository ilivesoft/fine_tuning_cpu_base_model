# Windows CPU 전용 빠른 시작 가이드 ⚡

## 🚀 5분 안에 시작하기

### 1단계: Python 설치 확인 (30초)
```cmd
python --version
```
✅ Python 3.8 이상이면 OK  
❌ 없다면 https://www.python.org/downloads/windows/ 에서 설치

---

### 2단계: 자동 설치 실행 (5분)
```cmd
# 프로젝트 폴더로 이동
cd /d D:\Projects\llama_cpu_project

# 자동 설치 스크립트 실행
install_windows_cpu.bat
```

설치되는 항목:
- ✅ 가상환경 생성
- ✅ PyTorch CPU 버전
- ✅ Transformers 및 필수 라이브러리

---

### 3단계: 시스템 확인 (10초)
```cmd
# 가상환경 활성화
llama_cpu_env\Scripts\activate

# 시스템 정보 확인
run_cpu_scripts.bat check
```

**확인 사항:**
- ✅ Python 버전
- ✅ CPU 코어 수
- ✅ RAM 크기
- ✅ PyTorch CPU 모드

---

### 4단계: 테스트 훈련 (5-10분)
```cmd
# 10 스텝만 빠르게 테스트
run_cpu_scripts.bat train_test
```

**예상 출력:**
```
🖥️ CPU 전용 Llama Fine-tuning Started
📦 Loading model and tokenizer on CPU...
📊 Loading and preprocessing data...
🎯 Starting training on CPU...
⚠️ CPU training is slow. Please be patient...
💾 Saving final model...
🎉 Training completed successfully!
```

---

### 5단계: 결과 확인 (1분)
```cmd
# 빠른 테스트
run_cpu_scripts.bat test
```

---

## 📋 전체 명령어 치트시트

### 기본 명령어
```cmd
# 시스템 확인
run_cpu_scripts.bat check

# 10 스텝 테스트
run_cpu_scripts.bat train_test

# 100 스텝 훈련
run_cpu_scripts.bat train_small

# 전체 훈련
run_cpu_scripts.bat train

# 대화형 추론
run_cpu_scripts.bat chat

# 빠른 테스트
run_cpu_scripts.bat test

# 단일 질문
run_cpu_scripts.bat ask "전세권이란?"

# 도움말
run_cpu_scripts.bat help
```

### Python 직접 실행
```cmd
# 훈련 (커스텀 옵션)
python main_train_cpu.py --max_steps 50 --save_steps 10

# 추론
python main_inference_cpu.py --model_path ./fine_tuned_model_cpu --interactive
```

---

## 🎯 추천 워크플로우

### 초보자 (30분)
```cmd
1. install_windows_cpu.bat          # 설치 (5분)
2. run_cpu_scripts.bat check        # 확인 (10초)
3. run_cpu_scripts.bat train_test   # 테스트 (10분)
4. run_cpu_scripts.bat test         # 결과 확인 (1분)
5. run_cpu_scripts.bat chat         # 대화 테스트 (자유)
```

### 중급자 (1-2시간)
```cmd
1. 자동 설치
2. run_cpu_scripts.bat train_small  # 100 스텝 (1시간)
3. run_cpu_scripts.bat chat         # 실전 테스트
4. 결과 분석 및 파라미터 조정
```

### 고급자 (수시간)
```cmd
1. 데이터셋 커스터마이징
2. config_cpu.py 파라미터 조정
3. 전체 훈련 실행
4. 성능 평가 및 반복
```

---

## ⚠️ 주의사항

### 시간 예상
- **10 스텝**: 5-10분 ⏱️
- **100 스텝**: 30-60분 ⏱️
- **1 에폭**: 수 시간 ⏱️⏱️⏱️

### 메모리 요구사항
- **최소**: RAM 32GB
- **권장**: RAM 64GB
- **가상 메모리**: 16-32GB 설정

### CPU 온도
- **정상**: 60-70°C
- **주의**: 70-80°C
- **위험**: 80°C 이상 (중단 권장)

---

## 🔧 문제 발생 시

### 메모리 부족
```cmd
# 가상 메모리 증가 (제어판)
제어판 → 시스템 → 고급 → 성능 → 가상 메모리
초기값: 16GB, 최대값: 32GB
```

### 너무 느림
```cmd
# 백그라운드 앱 종료
taskkill /f /im chrome.exe
taskkill /f /im firefox.exe

# 고성능 모드
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

### 패키지 오류
```cmd
# 가상환경 재생성
rmdir /s /q llama_cpu_env
python -m venv llama_cpu_env
llama_cpu_env\Scripts\activate
install_windows_cpu.bat
```

---

## 📞 자주 묻는 질문

### Q1: GPU 없이 정말 가능한가요?
**A**: 네! CPU만으로도 가능합니다. 다만 GPU보다 20-50배 느립니다.

### Q2: 최소 RAM이 얼마나 필요한가요?
**A**: 최소 32GB, 권장 64GB입니다.

### Q3: 훈련이 너무 오래 걸려요.
**A**: 처음에는 `train_test` (10 스텝)로 테스트하세요. 정상 작동 확인 후 확장하세요.

### Q4: 모델을 변경할 수 있나요?
**A**: `config_cpu.py`에서 `model_name`을 변경하세요:
```python
model_name: str = "skt/kogpt2-base-v2"  # 한국어 GPT2
# 또는
model_name: str = "beomi/kcbert-base"   # 한국어 BERT
```

### Q5: 데이터셋을 바꾸려면?
**A**: CSV 파일 형식은 동일하게 유지:
```csv
question,answer,category,difficulty
"질문","답변","카테고리","난이도"
```

---

## 📊 성능 벤치마크

### Intel i7-10세대 (8코어, 32GB RAM)
| 작업 | 소요 시간 |
|------|----------|
| 10 스텝 | 7분 |
| 100 스텝 | 45분 |
| 1 에폭 (60개) | 2.5시간 |

### AMD Ryzen 7 5800X (8코어, 64GB RAM)
| 작업 | 소요 시간 |
|------|----------|
| 10 스텝 | 4분 |
| 100 스텝 | 30분 |
| 1 에폭 (60개) | 1.5시간 |

---

## 🎓 다음 단계

### 학습 심화
1. `config_cpu.py` 파라미터 실험
2. 다양한 모델 테스트
3. 데이터셋 확장

### 성능 최적화
1. CPU 오버클럭 (전문가용)
2. RAM 증설
3. NVMe SSD 사용

### 프로젝트 확장
1. API 서버 구축
2. 웹 인터페이스 추가
3. 다른 도메인 적용

---

## 💡 팁

### 효율적인 개발
```cmd
# VS Code 터미널에서
1. Ctrl + ` (터미널 열기)
2. llama_cpu_env\Scripts\activate
3. 코드 수정 및 테스트 반복
```

### 로그 모니터링
```cmd
# 실시간 로그 확인 (PowerShell)
Get-Content training_cpu_*.log -Wait -Tail 10
```

### 자동 재시작 (긴 훈련용)
```cmd
# 오류 시 자동 재시작 (고급)
:loop
python main_train_cpu.py
if errorlevel 1 goto loop
```

---

## ✅ 체크리스트

설치 전:
- [ ] Python 3.8+ 설치됨
- [ ] RAM 32GB 이상
- [ ] 디스크 100GB 여유
- [ ] 관리자 권한 확보

설치 후:
- [ ] 가상환경 생성 완료
- [ ] PyTorch CPU 버전 설치
- [ ] `run_cpu_scripts.bat check` 성공
- [ ] `train_test` 정상 작동

---

이제 시작할 준비가 되었습니다! 🚀

문제가 있으면 `SETUP_WINDOWS_CPU.md`를 참고하세요.
