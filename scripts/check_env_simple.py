import sys
import torch

print("="*60)
print("🔍 환경 체크 (Simple)")
print("="*60)

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

if torch.backends.mps.is_available():
    print("✅ Apple M3 Pro GPU (MPS) 사용 가능")
    device = "mps"
else:
    print("💻 CPU 사용")
    device = "cpu"

# 패키지 버전 확인
packages = [
    'transformers',
    'datasets', 
    'pyarrow',
    'pandas',
    'numpy',
    'accelerate'
]

print("\n📦 패키지 상태:")
for pkg_name in packages:
    try:
        if pkg_name == 'pyarrow':
            import pyarrow
            print(f"  ✅ pyarrow: {pyarrow.__version__}")
        elif pkg_name == 'datasets':
            import datasets
            print(f"  ✅ datasets: {datasets.__version__}")
        elif pkg_name == 'transformers':
            import transformers
            print(f"  ✅ transformers: {transformers.__version__}")
        elif pkg_name == 'pandas':
            import pandas
            print(f"  ✅ pandas: {pandas.__version__}")
        elif pkg_name == 'numpy':
            import numpy
            print(f"  ✅ numpy: {numpy.__version__}")
        elif pkg_name == 'accelerate':
            import accelerate
            print(f"  ✅ accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"  ❌ {pkg_name}: 설치 필요 - {e}")

# 데이터 파일 확인
from pathlib import Path
print("\n📊 데이터 파일:")
for file in ['train.csv', 'validation.csv']:
    file_path = Path("data/processed") / file
    if file_path.exists():
        size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"  ✅ {file}: {size_mb:.2f} MB")

print("\n✅ 체크 완료!")
