import torch
import os
from pathlib import Path

class Config:
    # 경로 설정 (실제 구조에 맞게 수정)
    PROJECT_ROOT = Path.cwd()
    DATA_RAW = PROJECT_ROOT / "data" / "raw" / "143.인터페이스(자판-음성)별 고빈도 오류 교정 데이터" / "01-1.정식개방데이터"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # 디렉토리 생성
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 디바이스 설정 (M3 Pro)
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("✅ Using Apple M3 Pro GPU (MPS)")
    else:
        DEVICE = torch.device("cpu")
        print("💻 Using CPU")
    
    # 모델 설정
    MODEL_NAME = "gogamza/kobart-base-v2"
    MAX_LENGTH = 128
    
    # 훈련 설정
    BATCH_SIZE = 8  # M3 Pro 메모리에 적합
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.1
    
    # 데이터 설정
    TRAIN_SIZE = 0.9
    SEED = 42
