import os
from pathlib import Path
import json

def print_tree(directory, prefix="", max_depth=5, current_depth=0):
    """디렉토리 트리 구조 출력"""
    
    if current_depth >= max_depth:
        return
    
    # .DS_Store 제외
    items = [item for item in sorted(directory.iterdir()) if item.name != '.DS_Store']
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if item.is_dir():
            # 디렉토리
            print(f"{prefix}{current_prefix}📁 {item.name}/")
            
            # 하위 파일 개수 계산
            json_count = len(list(item.glob("**/*.json")))
            csv_count = len(list(item.glob("**/*.csv")))
            
            if json_count > 0 or csv_count > 0:
                stats_prefix = prefix + next_prefix
                print(f"{stats_prefix}   (JSON: {json_count}개, CSV: {csv_count}개)")
            
            # 재귀적으로 하위 디렉토리 출력
            print_tree(item, prefix + next_prefix, max_depth, current_depth + 1)
        else:
            # 파일
            if item.suffix in ['.json', '.csv']:
                icon = "📄" if item.suffix == '.json' else "📊"
                print(f"{prefix}{current_prefix}{icon} {item.name}")

def check_json_structure_sample():
    """JSON 파일 구조 샘플 확인"""
    base_path = Path("data/raw/143.인터페이스(자판-음성)별 고빈도 오류 교정 데이터/01-1.정식개방데이터")
    
    # 라벨링 데이터에서 JSON 찾기
    json_file = base_path / "Training" / "02.라벨링데이터" / "자주틀리는맞춤법오류.json"
    
    if json_file.exists():
        print("\n" + "="*60)
        print("📋 JSON 파일 구조 분석")
        print("="*60)
        print(f"파일: {json_file.name}\n")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # info 정보
            if 'info' in data:
                print("【info】")
                for key, value in data['info'].items():
                    print(f"  • {key}: {value}")
            
            # data 샘플
            if 'data' in data and len(data['data']) > 0:
                print(f"\n【data】 총 {len(data['data'])}개 항목")
                
                sample = data['data'][0]
                print("\n첫 번째 데이터 샘플:")
                
                if 'annotation' in sample:
                    ann = sample['annotation']
                    print(f"  ✏️ 원문: {ann.get('err_sentence', '')[:50]}...")
                    print(f"  ✅ 교정: {ann.get('cor_sentence', '')[:50]}...")

def main():
    base_path = Path("data/raw/143.인터페이스(자판-음성)별 고빈도 오류 교정 데이터/01-1.정식개방데이터")
    
    if not base_path.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {base_path}")
        return
    
    print("🗂️ 전체 디렉토리 구조")
    print("="*60)
    print_tree(base_path)
    
    # JSON 구조 샘플 확인
    check_json_structure_sample()

if __name__ == "__main__":
    main()
