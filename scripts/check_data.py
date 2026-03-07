from pathlib import Path
import zipfile

def check_data_structure():
    """데이터 구조 확인"""
    
    print("📁 데이터 구조 확인")
    print("="*60)
    
    # raw 폴더 확인
    raw_path = Path("data/raw")
    if not raw_path.exists():
        print("❌ data/raw 폴더가 없습니다!")
        return
    
    # 하위 폴더 확인
    for item in raw_path.iterdir():
        if item.is_dir():
            print(f"\n📂 {item.name}/")
            
            # 재귀적으로 하위 구조 출력
            for subitem in item.rglob("*"):
                if subitem.is_dir():
                    level = len(subitem.relative_to(item).parts)
                    indent = "  " * level
                    print(f"{indent}📁 {subitem.name}/")
                elif subitem.suffix == '.zip':
                    level = len(subitem.relative_to(item).parts)
                    indent = "  " * level
                    print(f"{indent}📦 {subitem.name}")
                    
                    # ZIP 파일 내용 샘플링
                    try:
                        with zipfile.ZipFile(subitem, 'r') as z:
                            files = z.namelist()[:3]
                            for f in files:
                                print(f"{indent}    - {f}")
                    except:
                        pass

if __name__ == "__main__":
    check_data_structure()
