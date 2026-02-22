import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_data(paths):
    """加载所有原始数据表"""
    # 定义统一日期格式
    date_format = '%d/%m/%Y %H:%M:%S'
    
    data = {
        'patients': pd.read_csv(paths['patients']),
        'admissions': pd.read_csv(
            paths['admissions'],
            parse_dates=['admittime', 'dischtime'],
            date_format=date_format,  # 替换为date_format参数
            dayfirst=True  # 明确指定日期开头为日
        ),
        'labevents': pd.read_csv(
            paths['labevents'],
            parse_dates=['charttime'],
            date_format=date_format,
            dayfirst=True
        ),
        'vitalsign': pd.read_csv(
            paths['vitalsign'],
            parse_dates=['charttime'],
            date_format=date_format,
            dayfirst=True
        ),
        'urine_output': pd.read_csv(
            paths['urine_output'],
            parse_dates=['charttime'],
            date_format=date_format,
            dayfirst=True
        ),
        'kdigo_stages': pd.read_csv(paths['kdigo_stages']),
        'creatinine_baseline': pd.read_csv(paths['creatinine_baseline']),
        'first_day_weight': pd.read_csv(paths['first_day_weight'])
    }

    # 添加时间解析验证
    for table in ['admissions', 'labevents', 'vitalsign', 'urine_output']:
        df = data[table]
        time_cols = [c for c in df.columns if df[c].dtype == 'datetime64[ns]']
        
        print(f"\n{table} 表时间解析验证:")
        for col in time_cols:
            na_count = df[col].isna().sum()
            print(f"{col} 列 - 解析失败数: {na_count}")
            if na_count > 0:
                # 输出前5个无法解析的原始值
                bad_samples = df.loc[df[col].isna(), col].head(5).to_list()
                print(f"示例无效值: {bad_samples}")

    return data


def build_base_cohort(data):
    """构建研究队列基础框架"""
    # 合并患者和入院信息
    base = data['admissions'][['subject_id', 'hadm_id', 'admittime', 'admission_type', 'race']]\
        .merge(data['patients'][['subject_id', 'anchor_age', 'gender']], on='subject_id')
    
    # 添加基线肌酐
    base = base.merge(
        data['creatinine_baseline'][['hadm_id', 'scr_baseline']], 
        on='hadm_id', how='left'
    )
    
    # 添加住院期间AKI标签
    aki_labels = data['kdigo_stages'].groupby('hadm_id')['aki_stage'].max().reset_index()
    base = base.merge(aki_labels, on='hadm_id', how='left')
    
    return base

def process_time_features(df, admittime):
    """处理时间相关特征"""
    df['hours_from_admit'] = (df['charttime'] - admittime).dt.total_seconds() / 3600
    return df[(df['hours_from_admit'] >= 0) & (df['hours_from_admit'] <= 24)]  # 取入院后24小时数据


def extract_lab_features(lab, base):
    """提取实验室指标特征"""
    # 筛选关键指标（需根据实际itemid映射）
    creat = lab[lab['itemid'].isin([50912])]  # 假设50912是肌酐的itemid
    
    # 合并到基础表
    lab_features = base[['hadm_id', 'admittime']].copy()
    
    # 计算肌酐特征
    creat_merged = lab_features.merge(creat, on='hadm_id', how='left')
    creat_merged = process_time_features(creat_merged, lab_features['admittime'])
    
    # 修复后的agg函数调用
    creat_features = creat_merged.groupby('hadm_id').agg(
        max_creat=('valuenum', 'max'),
        creat_slope=('valuenum', lambda x: np.polyfit(
            x.reset_index(drop=True).index,  # 使用重置后的索引
            x.values, 
            1
        )[0])
    )
    
    return creat_features


def extract_vital_features(vital, base):
    """提取生命体征特征"""
    vital_merged = base[['hadm_id', 'admittime']].merge(vital, on='hadm_id', how='left')
    vital_merged = process_time_features(vital_merged, base['admittime'])
    
    # 计算MAP
    vital_merged['map'] = vital_merged['dbp'] + (vital_merged['sbp'] - vital_merged['dbp'])/3
    
    vital_features = vital_merged.groupby('hadm_id').agg(
        min_map=('map', 'min'),
        max_hr=('heartrate', 'max'),
        resp_var=('resprate', 'std')
    )
    return vital_features

def process_urine_output(uo, weight, base):
    """处理尿量特征"""
    # 获取体重（假设已处理first_day_weight）
    uo_merged = base[['hadm_id', 'admittime']].merge(uo, on='hadm_id', how='left')
    uo_merged = process_time_features(uo_merged, base['admittime'])
    
    # 计算标准化尿量
    uo_merged = uo_merged.merge(weight, on='hadm_id')
    uo_merged['uo_ml_kg_hr'] = uo_merged['urineoutput'] / (uo_merged['weight'] * 6)
    
    # 计算6小时尿量
    uo_features = uo_merged.groupby('hadm_id').agg(
        uo_6hr=('urineoutput', lambda x: x.last('6h').sum()),
        uo_24hr=('urineoutput', 'sum')
    )
    return uo_features

def clean_data(df):
    """数据清洗处理"""
    # 异常值处理
    df['heartrate'] = df['heartrate'].clip(30, 250)
    df['resprate'] = df['resprate'].clip(5, 60)
    
    # 缺失值处理
    df['scr_baseline'] = df.groupby(['anchor_age', 'gender'])['scr_baseline']\
        .transform(lambda x: x.fillna(x.median()))
    
    # 分类变量编码
    race_freq = df['race'].value_counts(normalize=True)
    df['race_encoded'] = df['race'].map(race_freq)
    
    return df

def main(data_paths):
    # 加载数据
    data = load_data(data_paths)
    
    # 构建基础队列
    base = build_base_cohort(data)
    
    # 特征提取
    lab_features = extract_lab_features(data['labevents'], base)
    vital_features = extract_vital_features(data['vitalsign'], base)
    uo_features = process_urine_output(
        data['urine_output'], 
        data['first_day_weight'], 
        base
    )
    
    # 合并所有特征
    final_df = base.merge(lab_features, on='hadm_id', how='left')\
                   .merge(vital_features, on='hadm_id', how='left')\
                   .merge(uo_features, on='hadm_id', how='left')
    
    # 数据清洗
    final_df = clean_data(final_df)
    
    # 特征选择
    features = [
        'anchor_age', 'gender', 'race_encoded', 'scr_baseline',
        'max_creat', 'creat_slope', 'min_map', 'max_hr',
        'uo_6hr', 'uo_24hr', 'aki_stage'
    ]
    final_df = final_df[features].dropna()
    
    # 转换为HuggingFace格式
    hf_dataset = Dataset.from_pandas(final_df)
    
    # 数据集拆分
    train_test = hf_dataset.train_test_split(test_size=0.2, stratify_by_column='aki_stage')
    
    return train_test

if __name__ == "__main__":
    data_paths = {
        'patients': 'patients.csv',
        'admissions': 'admissions.csv',
        'labevents': 'labevents.csv',
        'vitalsign': 'vitalsign.csv',
        'urine_output': 'urine_output.csv',
        'kdigo_stages': 'kdigo_stages.csv',
        'creatinine_baseline': 'creatinine_baseline.csv',
        'first_day_weight': 'first_day_weight.csv'
    }
    
    dataset = main(data_paths)
    dataset.save_to_disk("processed_aki_dataset")
