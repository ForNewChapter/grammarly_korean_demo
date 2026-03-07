import json
import zipfile
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import os
from config import Config

class AIHubDataProcessor:
    def __init__(self):
        self.config = Config()
        self.data_path = self.config.DATA_RAW
        
        # 경로 확인
        if not self.data_path.exists():
            print(f"❌ 경로를 찾을 수 없습니다: {self.data_path}")
            print(f"현재 작업 디렉토리: {Path.cwd()}")
            print("\n사용 가능한 경로:")
            for p in Path("data/raw").glob("*"):
                print(f"  - {p}")
            raise FileNotFoundError(f"데이터 경로를 확인하세요: {self.data_path}")
    
    def extract_json_from_zip(self, zip_path: Path) -> List[Dict]:
        """ZIP 파일에서 JSON 데이터 추출"""
        data_list = []
        
        print(f"  📦 Processing: {zip_path.name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # ZIP 파일 내용 확인
                file_list = z.namelist()
                json_files = [f for f in file_list if f.endswith('.json')]
                
                if not json_files:
                    print(f"    ⚠️ No JSON files in {zip_path.name}")
                    # JSON이 없으면 다른 형식 확인
                    for fname in file_list[:5]:  # 처음 5개만 출력
                        print(f"      - {fname}")
                
                for filename in json_files:
                    with z.open(filename) as f:
                        content = f.read().decode('utf-8')
                        try:
                            data = json.loads(content)
                            if isinstance(data, list):
                                data_list.extend(data)
                            else:
                                data_list.append(data)
                            print(f"    ✅ Loaded {len(data) if isinstance(data, list) else 1} items from {filename}")
                        except json.JSONDecodeError as e:
                            print(f"    ❌ Error reading {filename}: {e}")
                        except Exception as e:
                            print(f"    ❌ Unexpected error: {e}")
                            
        except zipfile.BadZipFile:
            print(f"    ❌ {zip_path.name} is not a valid zip file")
        except Exception as e:
            print(f"    ❌ Error processing {zip_path.name}: {e}")
        
        return data_list
    
    def extract_data_from_item(self, item: Dict) -> Optional[Dict]:
        """다양한 JSON 구조에서 원문-교정문 추출"""
        # 가능한 키 조합들
        key_pairs = [
            ('original', 'corrected'),
            ('source_sentence', 'target_sentence'),
            ('input', 'output'),
            ('text', 'correction'),
            ('원문', '정답'),
            ('source', 'target'),
            ('error_sentence', 'correct_sentence')
        ]
        
        for orig_key, corr_key in key_pairs:
            if orig_key in item and corr_key in item:
                return {
                    'original': str(item[orig_key]).strip(),
                    'corrected': str(item[corr_key]).strip()
                }
        
        # 키가 없으면 전체 구조 확인
        if item:
            print(f"    ⚠️ Unknown structure. Keys: {list(item.keys())[:5]}")
        
        return None
    
    def process_all_data(self):
        """모든 데이터 처리"""
        all_data = []
        
        print("\n" + "="*60)
        print("📊 데이터 처리 시작")
        print("="*60)
        
        # Training 데이터 처리
        train_label_path = self.data_path / "Training" / "02.라벨링데이터"
        
        if not train_label_path.exists():
            print(f"❌ Training 라벨링 경로를 찾을 수 없습니다: {train_label_path}")
            # 대체 경로 시도
            train_label_path = self.data_path / "Training" / "라벨링데이터"
            if not train_label_path.exists():
                raise FileNotFoundError(f"라벨링 데이터 폴더를 찾을 수 없습니다")
        
        print(f"\n📁 Training 데이터 처리")
        print(f"   경로: {train_label_path}")
        
        # 모든 ZIP 파일 찾기
        zip_files = list(train_label_path.glob("*.zip"))
        print(f"   발견된 ZIP 파일: {len(zip_files)}개")
        
        for zip_file in zip_files:
            data = self.extract_json_from_zip(zip_file)
            
            for item in data:
                extracted = self.extract_data_from_item(item)
                if extracted:
                    extracted['split'] = 'train'
                    extracted['source_file'] = zip_file.name
                    all_data.append(extracted)
        
        print(f"\n✅ Training 데이터: {len([d for d in all_data if d['split']=='train'])}개")
        
        # Validation 데이터 처리
        val_label_path = self.data_path / "Validation" / "02.라벨링데이터"
        
        if not val_label_path.exists():
            print(f"⚠️ Validation 경로를 찾을 수 없습니다: {val_label_path}")
            val_label_path = self.data_path / "Validation" / "라벨링데이터"
        
        if val_label_path.exists():
            print(f"\n📁 Validation 데이터 처리")
            print(f"   경로: {val_label_path}")
            
            zip_files = list(val_label_path.glob("*.zip"))
            print(f"   발견된 ZIP 파일: {len(zip_files)}개")
            
            for zip_file in zip_files:
                data = self.extract_json_from_zip(zip_file)
                
                for item in data:
                    extracted = self.extract_data_from_item(item)
                    if extracted:
                        extracted['split'] = 'validation'
                        extracted['source_file'] = zip_file.name
                        all_data.append(extracted)
            
            print(f"\n✅ Validation 데이터: {len([d for d in all_data if d['split']=='validation'])}개")
        
        if not all_data:
            print("\n❌ 데이터를 찾을 수 없습니다!")
            print("\n디버깅 정보:")
            print("1. ZIP 파일 내용 확인이 필요합니다")
            print("2. 수동으로 ZIP 파일을 열어 JSON 구조를 확인하세요")
            return None
        
        # DataFrame 생성
        df = pd.DataFrame(all_data)
        
        # 데이터 정제
        print("\n🧹 데이터 정제 중...")
        original_len = len(df)
        
        # 중복 제거
        df = df.drop_duplicates(subset=['original', 'corrected'])
        
        # 빈 값 제거
        df = df.dropna(subset=['original', 'corrected'])
        
        # 동일한 원문-교정문 제거
        df = df[df['original'] != df['corrected']]
        
        # 너무 짧거나 긴 문장 제거
        df = df[df['original'].str.len() > 1]
        df = df[df['original'].str.len() <= 200]
        
        print(f"   정제 전: {original_len}개 → 정제 후: {len(df)}개")
        
        # 통계 출력
        print("\n📊 최종 데이터 통계")
        print("="*40)
        print(f"총 데이터: {len(df)}개")
        print(f"Train: {len(df[df['split']=='train'])}개")
        print(f"Validation: {len(df[df['split']=='validation'])}개")
        
        if 'source_file' in df.columns:
            print("\n파일별 데이터 수:")
            for fname in df['source_file'].unique():
                count = len(df[df['source_file']==fname])
                print(f"  - {fname}: {count}개")
        
        # 샘플 출력
        print("\n📝 데이터 샘플 (처음 3개):")
        print("-"*40)
        for idx, row in df.head(3).iterrows():
            print(f"원문: {row['original'][:50]}...")
            print(f"교정: {row['corrected'][:50]}...")
            print(f"구분: {row['split']}")
            print("-"*40)
        
        # 저장
        save_path = self.config.DATA_PROCESSED / "processed_data.csv"
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n�� 데이터 저장 완료: {save_path}")
        
        return df

if __name__ == "__main__":
    try:
        processor = AIHubDataProcessor()
        df = processor.process_all_data()
        
        if df is not None and len(df) > 0:
            print("\n✅ 데이터 처리 성공!")
        else:
            print("\n❌ 데이터 처리 실패!")
            print("\n해결 방법:")
            print("1. ZIP 파일을 수동으로 열어 내용 확인")
            print("2. JSON 파일의 구조 확인")
            print("3. 필요시 스크립트의 extract_data_from_item 함수 수정")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
EOFcat > scripts/data_processor.py << 'EOF'
import json
import zipfile
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import os
from config import Config

class AIHubDataProcessor:
    def __init__(self):
        self.config = Config()
        self.data_path = self.config.DATA_RAW
        
        # 경로 확인
        if not self.data_path.exists():
            print(f"❌ 경로를 찾을 수 없습니다: {self.data_path}")
            print(f"현재 작업 디렉토리: {Path.cwd()}")
            print("\n사용 가능한 경로:")
            for p in Path("data/raw").glob("*"):
                print(f"  - {p}")
            raise FileNotFoundError(f"데이터 경로를 확인하세요: {self.data_path}")
    
    def extract_json_from_zip(self, zip_path: Path) -> List[Dict]:
        """ZIP 파일에서 JSON 데이터 추출"""
        data_list = []
        
        print(f"  📦 Processing: {zip_path.name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # ZIP 파일 내용 확인
                file_list = z.namelist()
                json_files = [f for f in file_list if f.endswith('.json')]
                
                if not json_files:
                    print(f"    ⚠️ No JSON files in {zip_path.name}")
                    # JSON이 없으면 다른 형식 확인
                    for fname in file_list[:5]:  # 처음 5개만 출력
                        print(f"      - {fname}")
                
                for filename in json_files:
                    with z.open(filename) as f:
                        content = f.read().decode('utf-8')
                        try:
                            data = json.loads(content)
                            if isinstance(data, list):
                                data_list.extend(data)
                            else:
                                data_list.append(data)
                            print(f"    ✅ Loaded {len(data) if isinstance(data, list) else 1} items from {filename}")
                        except json.JSONDecodeError as e:
                            print(f"    ❌ Error reading {filename}: {e}")
                        except Exception as e:
                            print(f"    ❌ Unexpected error: {e}")
                            
        except zipfile.BadZipFile:
            print(f"    ❌ {zip_path.name} is not a valid zip file")
        except Exception as e:
            print(f"    ❌ Error processing {zip_path.name}: {e}")
        
        return data_list
    
    def extract_data_from_item(self, item: Dict) -> Optional[Dict]:
        """다양한 JSON 구조에서 원문-교정문 추출"""
        # 가능한 키 조합들
        key_pairs = [
            ('original', 'corrected'),
            ('source_sentence', 'target_sentence'),
            ('input', 'output'),
            ('text', 'correction'),
            ('원문', '정답'),
            ('source', 'target'),
            ('error_sentence', 'correct_sentence')
        ]
        
        for orig_key, corr_key in key_pairs:
            if orig_key in item and corr_key in item:
                return {
                    'original': str(item[orig_key]).strip(),
                    'corrected': str(item[corr_key]).strip()
                }
        
        # 키가 없으면 전체 구조 확인
        if item:
            print(f"    ⚠️ Unknown structure. Keys: {list(item.keys())[:5]}")
        
        return None
    
    def process_all_data(self):
        """모든 데이터 처리"""
        all_data = []
        
        print("\n" + "="*60)
        print("📊 데이터 처리 시작")
        print("="*60)
        
        # Training 데이터 처리
        train_label_path = self.data_path / "Training" / "02.라벨링데이터"
        
        if not train_label_path.exists():
            print(f"❌ Training 라벨링 경로를 찾을 수 없습니다: {train_label_path}")
            # 대체 경로 시도
            train_label_path = self.data_path / "Training" / "라벨링데이터"
            if not train_label_path.exists():
                raise FileNotFoundError(f"라벨링 데이터 폴더를 찾을 수 없습니다")
        
        print(f"\n📁 Training 데이터 처리")
        print(f"   경로: {train_label_path}")
        
        # 모든 ZIP 파일 찾기
        zip_files = list(train_label_path.glob("*.zip"))
        print(f"   발견된 ZIP 파일: {len(zip_files)}개")
        
        for zip_file in zip_files:
            data = self.extract_json_from_zip(zip_file)
            
            for item in data:
                extracted = self.extract_data_from_item(item)
                if extracted:
                    extracted['split'] = 'train'
                    extracted['source_file'] = zip_file.name
                    all_data.append(extracted)
        
        print(f"\n✅ Training 데이터: {len([d for d in all_data if d['split']=='train'])}개")
        
        # Validation 데이터 처리
        val_label_path = self.data_path / "Validation" / "02.라벨링데이터"
        
        if not val_label_path.exists():
            print(f"⚠️ Validation 경로를 찾을 수 없습니다: {val_label_path}")
            val_label_path = self.data_path / "Validation" / "라벨링데이터"
        
        if val_label_path.exists():
            print(f"\n📁 Validation 데이터 처리")
            print(f"   경로: {val_label_path}")
            
            zip_files = list(val_label_path.glob("*.zip"))
            print(f"   발견된 ZIP 파일: {len(zip_files)}개")
            
            for zip_file in zip_files:
                data = self.extract_json_from_zip(zip_file)
                
                for item in data:
                    extracted = self.extract_data_from_item(item)
                    if extracted:
                        extracted['split'] = 'validation'
                        extracted['source_file'] = zip_file.name
                        all_data.append(extracted)
            
            print(f"\n✅ Validation 데이터: {len([d for d in all_data if d['split']=='validation'])}개")
        
        if not all_data:
            print("\n❌ 데이터를 찾을 수 없습니다!")
            print("\n디버깅 정보:")
            print("1. ZIP 파일 내용 확인이 필요합니다")
            print("2. 수동으로 ZIP 파일을 열어 JSON 구조를 확인하세요")
            return None
        
        # DataFrame 생성
        df = pd.DataFrame(all_data)
        
        # 데이터 정제
        print("\n🧹 데이터 정제 중...")
        original_len = len(df)
        
        # 중복 제거
        df = df.drop_duplicates(subset=['original', 'corrected'])
        
        # 빈 값 제거
        df = df.dropna(subset=['original', 'corrected'])
        
        # 동일한 원문-교정문 제거
        df = df[df['original'] != df['corrected']]
        
        # 너무 짧거나 긴 문장 제거
        df = df[df['original'].str.len() > 1]
        df = df[df['original'].str.len() <= 200]
        
        print(f"   정제 전: {original_len}개 → 정제 후: {len(df)}개")
        
        # 통계 출력
        print("\n📊 최종 데이터 통계")
        print("="*40)
        print(f"총 데이터: {len(df)}개")
        print(f"Train: {len(df[df['split']=='train'])}개")
        print(f"Validation: {len(df[df['split']=='validation'])}개")
        
        if 'source_file' in df.columns:
            print("\n파일별 데이터 수:")
            for fname in df['source_file'].unique():
                count = len(df[df['source_file']==fname])
                print(f"  - {fname}: {count}개")
        
        # 샘플 출력
        print("\n📝 데이터 샘플 (처음 3개):")
        print("-"*40)
        for idx, row in df.head(3).iterrows():
            print(f"원문: {row['original'][:50]}...")
            print(f"교정: {row['corrected'][:50]}...")
            print(f"구분: {row['split']}")
            print("-"*40)
        
        # 저장
        save_path = self.config.DATA_PROCESSED / "processed_data.csv"
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n�� 데이터 저장 완료: {save_path}")
        
        return df

if __name__ == "__main__":
    try:
        processor = AIHubDataProcessor()
        df = processor.process_all_data()
        
        if df is not None and len(df) > 0:
            print("\n✅ 데이터 처리 성공!")
        else:
            print("\n❌ 데이터 처리 실패!")
            print("\n해결 방법:")
            print("1. ZIP 파일을 수동으로 열어 내용 확인")
            print("2. JSON 파일의 구조 확인")
            print("3. 필요시 스크립트의 extract_data_from_item 함수 수정")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
