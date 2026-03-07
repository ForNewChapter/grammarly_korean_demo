import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import os

class AIHubDataProcessor:
    def __init__(self):
        self.base_path = Path("data/raw/143.인터페이스(자판-음성)별 고빈도 오류 교정 데이터/01-1.정식개방데이터")
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {self.base_path}")
    
    def read_json_file(self, file_path: Path) -> List[Dict]:
        """JSON 파일 읽기"""
        data_list = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    data_list.extend(data)
                else:
                    data_list.append(data)
        except Exception as e:
            print(f"    ❌ Error reading {file_path.name}: {e}")
        return data_list
    
    def read_csv_file(self, file_path: Path) -> List[Dict]:
        """CSV 파일 읽기"""
        data_list = []
        try:
            # 다양한 인코딩 시도
            encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            data_list.append(dict(row))
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if encoding == encodings[-1]:
                        print(f"    ❌ Error reading {file_path.name}: {e}")
        except Exception as e:
            print(f"    ❌ Error reading {file_path.name}: {e}")
        
        return data_list
    
    def extract_data_from_item(self, item: Dict) -> Optional[Dict]:
        """다양한 데이터 구조에서 원문-교정문 추출"""
        # 가능한 키 조합들
        key_pairs = [
            ('original', 'corrected'),
            ('source_sentence', 'target_sentence'),
            ('input', 'output'),
            ('text', 'correction'),
            ('원문', '정답'),
            ('source', 'target'),
            ('error_sentence', 'correct_sentence'),
            ('문장', '정답문장'),
            ('오류문장', '교정문장'),
            ('before', 'after')
        ]
        
        for orig_key, corr_key in key_pairs:
            if orig_key in item and corr_key in item:
                original = str(item[orig_key]).strip()
                corrected = str(item[corr_key]).strip()
                
                # 유효한 데이터인지 확인
                if original and corrected and original != corrected:
                    return {
                        'original': original,
                        'corrected': corrected
                    }
        
        return None
    
    def process_directory(self, dir_path: Path, split: str) -> List[Dict]:
        """디렉토리 내 모든 파일 처리"""
        all_data = []
        
        # 라벨링데이터 폴더 찾기
        label_dirs = [
            dir_path / "02.라벨링데이터",
            dir_path / "라벨링데이터",
            dir_path / "02_라벨링데이터"
        ]
        
        label_dir = None
        for ld in label_dirs:
            if ld.exists():
                label_dir = ld
                break
        
        if not label_dir:
            print(f"  ⚠️ 라벨링데이터 폴더를 찾을 수 없습니다")
            # 전체 디렉토리에서 JSON/CSV 찾기
            label_dir = dir_path
        
        print(f"  📁 처리 중: {label_dir}")
        
        # 모든 하위 디렉토리 포함하여 파일 찾기
        json_files = list(label_dir.glob("**/*.json"))
        csv_files = list(label_dir.glob("**/*.csv"))
        
        print(f"    - JSON 파일: {len(json_files)}개")
        print(f"    - CSV 파일: {len(csv_files)}개")
        
        # JSON 파일 처리
        for json_file in json_files:
            print(f"    📄 {json_file.name}")
            data = self.read_json_file(json_file)
            
            for item in data:
                extracted = self.extract_data_from_item(item)
                if extracted:
                    extracted['split'] = split
                    extracted['source_file'] = json_file.name
                    all_data.append(extracted)
        
        # CSV 파일 처리
        for csv_file in csv_files:
            print(f"    📄 {csv_file.name}")
            data = self.read_csv_file(csv_file)
            
            for item in data:
                extracted = self.extract_data_from_item(item)
                if extracted:
                    extracted['split'] = split
                    extracted['source_file'] = csv_file.name
                    all_data.append(extracted)
        
        return all_data
    
    def process_all_data(self):
        """모든 데이터 처리"""
        all_data = []
        
        print("\n" + "="*60)
        print("📊 데이터 처리 시작")
        print("="*60)
        
        # Training 데이터 처리
        train_path = self.base_path / "Training"
        if train_path.exists():
            print("\n📂 Training 데이터 처리")
            train_data = self.process_directory(train_path, 'train')
            all_data.extend(train_data)
            print(f"  ✅ 수집된 데이터: {len(train_data)}개")
        
        # Validation 데이터 처리
        val_path = self.base_path / "Validation"
        if val_path.exists():
            print("\n📂 Validation 데이터 처리")
            val_data = self.process_directory(val_path, 'validation')
            all_data.extend(val_data)
            print(f"  ✅ 수집된 데이터: {len(val_data)}개")
        
        if not all_data:
            print("\n❌ 데이터를 찾을 수 없습니다!")
            return None
        
        # DataFrame 생성
        df = pd.DataFrame(all_data)
        
        # 데이터 정제
        print("\n🧹 데이터 정제 중...")
        original_len = len(df)
        
        # 중복 제거
        df = df.drop_duplicates(subset=['original', 'corrected'])
        
        # 너무 짧거나 긴 문장 제거
        df = df[df['original'].str.len() > 1]
        df = df[df['original'].str.len() <= 200]
        
        print(f"  정제 전: {original_len}개 → 정제 후: {len(df)}개")
        
        # 통계 출력
        print("\n📊 최종 데이터 통계")
        print("="*40)
        print(f"총 데이터: {len(df)}개")
        print(f"Train: {len(df[df['split']=='train'])}개")
        print(f"Validation: {len(df[df['split']=='validation'])}개")
        
        # 샘플 출력
        print("\n📝 데이터 샘플 (처음 3개):")
        print("-"*40)
        for idx, row in df.head(3).iterrows():
            print(f"원문: {row['original'][:50]}")
            print(f"교정: {row['corrected'][:50]}")
            print("-"*40)
        
        # 저장
        save_path = self.processed_path / "processed_data.csv"
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 데이터 저장 완료: {save_path}")
        
        return df

if __name__ == "__main__":
    try:
        processor = AIHubDataProcessor()
        df = processor.process_all_data()
        
        if df is not None and len(df) > 0:
            print("\n✅ 데이터 처리 성공!")
        else:
            print("\n❌ 데이터 처리 실패!")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
