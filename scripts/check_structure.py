from pathlib import Path
import json
import csv

def check_file_structure():
    """실제 파일 구조 확인"""
    
    print("📁 파일 구조 확인")
    print("="*60)
    
    base_path = Path("data/raw/143.인터페이스(자판-음성)별 고빈도 오류 교정 데이터/01-1.정식개방데이터")
    
    if not base_path.exists():
        print(f"❌ 경로가 없습니다: {base_path}")
        return
    
    # Training 폴더 확인
    train_path = base_path / "Training"
    if train_path.exists():
        print("\n📂 Training/")
        for sub in train_path.iterdir():
            print(f"  📁 {sub.name}/")
            
            # JSON 파일 확인
            json_files = list(sub.glob("**/*.json"))
            csv_files = list(sub.glob("**/*.csv"))
            
            print(f"    JSON 파일: {len(json_files)}개")
            print(f"    CSV 파일: {len(csv_files)}개")
            
            # 샘플 파일 구조 확인
            if json_files:
                sample_json = json_files[0]
                print(f"    📄 샘플 JSON: {sample_json.name}")
                try:
                    with open(sample_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            print(f"      - 데이터 수: {len(data)}개")
                            print(f"      - 키: {list(data[0].keys())}")
                        elif isinstance(data, dict):
                            print(f"      - 키: {list(data.keys())}")
                except Exception as e:
                    print(f"      ❌ 읽기 오류: {e}")
            
            if csv_files:
                sample_csv = csv_files[0]
                print(f"    📄 샘플 CSV: {sample_csv.name}")
                try:
                    with open(sample_csv, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        first_row = next(reader, None)
                        if first_row:
                            print(f"      - 컬럼: {list(first_row.keys())}")
                except Exception as e:
                    print(f"      ❌ 읽기 오류: {e}")
    
    # Validation 폴더도 동일하게 확인
    val_path = base_path / "Validation"
    if val_path.exists():
        print("\n📂 Validation/")
        for sub in val_path.iterdir():
            print(f"  📁 {sub.name}/")
            json_files = list(sub.glob("**/*.json"))
            csv_files = list(sub.glob("**/*.csv"))
            print(f"    JSON 파일: {len(json_files)}개")
            print(f"    CSV 파일: {len(csv_files)}개")

if __name__ == "__main__":
    check_file_structure()
