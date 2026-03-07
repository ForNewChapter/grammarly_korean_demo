import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import os
from tqdm import tqdm

class AIHubDataProcessor:
    def __init__(self):
        self.base_path = Path("data/raw/143.인터페이스(자판-음성)별 고빈도 오류 교정 데이터/01-1.정식개방데이터")
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {self.base_path}")
    
    def process_json_file(self, file_path: Path) -> List[Dict]:
        """JSON 파일 처리 (AI Hub 구조)"""
        processed_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # AI Hub JSON 구조: info와 data 키
            if 'data' in json_data:
                for item in json_data['data']:
                    # annotation 키에서 원문과 교정문 추출
                    if 'annotation' in item:
                        ann = item['annotation']
                        
                        # 오류 문장과 교정 문장
                        err_sentence = ann.get('err_sentence', '')
                        cor_sentence = ann.get('cor_sentence', '')
                        
                        if err_sentence and cor_sentence and err_sentence != cor_sentence:
                            processed_data.append({
                                'original': err_sentence,
                                'corrected': cor_sentence,
                                'error_type': file_path.stem,  # 파일명을 오류 타입으로 사용
                                'source': 'json'
                            })
            
            print(f"    ✅ {file_path.name}: {len(processed_data)}개 데이터 추출")
            
        except Exception as e:
            print(f"    ❌ {file_path.name} 처리 오류: {e}")
        
        return processed_data
    
    def process_csv_file(self, file_path: Path) -> List[Dict]:
        """CSV 파일 처리 (원천 데이터)"""
        processed_data = []
        
        # CSV는 원천 데이터이므로 교정 데이터가 없을 수 있음
        # 일단 건너뛰거나 다른 방식으로 처리
        
        return processed_data
    
    def process_directory(self, dir_path: Path, data_type: str) -> List[Dict]:
        """디렉토리 처리"""
        all_data = []
        
        # 라벨링 데이터 폴더 (JSON 파일들)
        label_dir = dir_path / "02.라벨링데이터"
        if label_dir.exists():
            print(f"\n  �� {data_type} - 라벨링데이터 처리")
            
            json_files = [f for f in label_dir.glob("*.json") if f.name != '.DS_Store']
            
            for json_file in json_files:
                data = self.process_json_file(json_file)
                for item in data:
                    item['split'] = data_type.lower()
                all_data.extend(data)
        
        return all_data
    
    def process_all_data(self):
        """모든 데이터 처리"""
        all_data = []
        
        print("\n" + "="*60)
        print("📊 AI Hub 맞춤법 교정 데이터 처리")
        print("="*60)
        
        # Training 데이터 처리
        train_path = self.base_path / "Training"
        if train_path.exists():
            train_data = self.process_directory(train_path, 'train')
            all_data.extend(train_data)
            print(f"  ✅ Training 총합: {len(train_data)}개")
        
        # Validation 데이터 처리
        val_path = self.base_path / "Validation"
        if val_path.exists():
            val_data = self.process_directory(val_path, 'validation')
            all_data.extend(val_data)
            print(f"  ✅ Validation 총합: {len(val_data)}개")
        
        if not all_data:
            print("\n❌ 데이터를 찾을 수 없습니다!")
            return None
        
        # DataFrame 생성
        df = pd.DataFrame(all_data)
        
        # 데이터 정제
        print("\n🧹 데이터 정제")
        print("-"*40)
        
        original_len = len(df)
        
        # 중복 제거
        df = df.drop_duplicates(subset=['original', 'corrected'])
        
        # 너무 짧거나 긴 문장 제거
        df = df[df['original'].str.len() > 1]
        df = df[df['original'].str.len() <= 300]
        
        # 공백 정규화
        df['original'] = df['original'].str.strip()
        df['corrected'] = df['corrected'].str.strip()
        
        print(f"  • 원본: {original_len}개 → 정제: {len(df)}개")
        print(f"  • 제거: {original_len - len(df)}개")
        
        # 통계 출력
        print("\n📊 최종 데이터 통계")
        print("-"*40)
        print(f"  • 총 데이터: {len(df):,}개")
        print(f"  • Train: {len(df[df['split']=='train']):,}개")
        print(f"  • Validation: {len(df[df['split']=='validation']):,}개")
        
        if 'error_type' in df.columns:
            print("\n  오류 타입별 분포:")
            for error_type in df['error_type'].unique():
                count = len(df[df['error_type']==error_type])
                pct = count / len(df) * 100
                print(f"    - {error_type}: {count:,}개 ({pct:.1f}%)")
        
        # 샘플 출력
        print("\n📝 데이터 샘플")
        print("-"*40)
        for idx, row in df.head(5).iterrows():
            print(f"\n[{idx+1}]")
            print(f"  원문: {row['original'][:60]}...")
            print(f"  교정: {row['corrected'][:60]}...")
            print(f"  타입: {row.get('error_type', 'N/A')}")
        
        # 저장
        save_path = self.processed_path / "processed_data.csv"
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 데이터 저장 완료")
        print(f"  경로: {save_path}")
        print(f"  크기: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 추가 저장 (train/val 분리)
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'validation']
        
        train_df.to_csv(self.processed_path / "train.csv", index=False, encoding='utf-8-sig')
        val_df.to_csv(self.processed_path / "validation.csv", index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 처리 완료!")
        print(f"  • train.csv: {len(train_df):,}개")
        print(f"  • validation.csv: {len(val_df):,}개")
        
        return df

if __name__ == "__main__":
    try:
        processor = AIHubDataProcessor()
        df = processor.process_all_data()
        
        if df is not None and len(df) > 0:
            print("\n🎉 데이터 처리 성공!")
            print("\n다음 단계: python scripts/train_model.py")
        else:
            print("\n❌ 데이터 처리 실패!")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
