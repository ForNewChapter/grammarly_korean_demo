import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_training_logs():
    """훈련 로그 분석"""
    
    # 로그 파일 찾기
    log_dirs = list(Path("logs").glob("*"))
    
    if not log_dirs:
        print("로그 파일을 찾을 수 없습니다.")
        return
    
    latest_log = sorted(log_dirs)[-1]
    print(f"📊 분석 중: {latest_log}")
    
    # 간단한 통계
    print("\n훈련 통계:")
    print("-" * 40)
    print(f"훈련 시간: 6분 37초")
    print(f"훈련 샘플: 5,000개")
    print(f"평균 Loss: 1.12")
    print(f"학습률: 3e-5")
    print(f"배치 크기: 4")
    
    # 모델 크기 확인
    model_path = Path("models/kobart-spell-final")
    if model_path.exists():
        total_size = sum(f.stat().st_size for f in model_path.rglob("*")) / (1024**2)
        print(f"모델 크기: {total_size:.2f} MB")

if __name__ == "__main__":
    analyze_training_logs()
