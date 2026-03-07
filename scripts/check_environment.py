import sys
import torch
import platform

def check_environment():
    print("="*60)
    print("🔍 환경 체크")
    print("="*60)
    
    # Python 버전
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # PyTorch
    print(f"\nPyTorch: {torch.__version__}")
    
    # Device 확인
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✅ Apple Silicon GPU (MPS) 사용 가능")
        # MPS 메모리 확인
        print(f"   Device: {device}")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA GPU 사용 가능")
    else:
        device = "cpu"
        print(f"💻 CPU 사용")
    
    print(f"\n권장 디바이스: {device}")
    
    # 필수 패키지 확인
    packages = {
        'transformers': None,
        'datasets': None,
        'accelerate': None,
        'pandas': None,
        'numpy': None,
        'tqdm': None
    }
    
    print("\n📦 패키지 상태:")
    for package in packages:
        try:
            mod = __import__(package)
            packages[package] = mod.__version__
            print(f"  ✅ {package}: {packages[package]}")
        except ImportError:
            print(f"  ❌ {package}: 설치 필요")
    
    # 데이터 확인
    from pathlib import Path
    data_path = Path("data/processed")
    
    print("\n📊 데이터 파일:")
    for file in ['train.csv', 'validation.csv']:
        file_path = data_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  ✅ {file}: {size_mb:.2f} MB")
        else:
            print(f"  ❌ {file}: 없음")
    
    return device

if __name__ == "__main__":
    device = check_environment()
    print("\n준비 완료! 훈련을 시작할 수 있습니다.")
