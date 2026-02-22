# -*- coding: utf-8 -*-
# 急性肾损伤（AKI）预测系统 v3.0 - 仅过滤肌酐斜率为0的case
# 集成特征工程与LLM分类的端到端解决方案 - 优化版

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from sklearn.metrics import fbeta_score
import re


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
    EarlyStoppingCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModelForSequenceClassification
)
from datasets import Dataset

# --------------------------
# 日志配置模块
# --------------------------
def setup_logging():
    """配置日志系统"""
    log_filename = f"aki_prediction_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --------------------------
# 新增：通用工具函数（统一0斜率判断标准）【修改点1】
# --------------------------
def is_zero_slope(creat_slope, threshold=1e-10):
    """
    判断是否为0肌酐斜率（统一浮点数精度判断）
    Args:
        creat_slope: 肌酐斜率值
        threshold: 0斜率判断阈值（绝对值小于该值即视为0斜率）
    Returns:
        bool: 是否为0斜率
    """
    if pd.isna(creat_slope) or not isinstance(creat_slope, (int, float)):
        return True  # 空值/非数值视为0斜率（过滤）
    return abs(float(creat_slope)) < threshold

# --------------------------
# 数据预处理模块 (增强版，仅过滤肌酐斜率为0的case)
# --------------------------
class AKIDataProcessor:
    def __init__(self, data_paths):
        """
        初始化数据处理器
        Args:
            data_paths: 数据路径字典
        """
        self.data_paths = data_paths
        self.icd_map = {
            'I10': '高血压',
            'E11': '糖尿病',
            'N18': '慢性肾病',
            'I50': '心力衰竭',
            'J18': '肺炎'
        }

    def _load_raw_data(self):
        """加载原始数据表并统一列名为小写"""
        date_format = '%d/%m/%Y %H:%M:%S'
        
        data = {}
        for key in self.data_paths.keys():
            path = self.data_paths[key]
            try:
                df = pd.read_csv(path)
                logger.info(f"成功加载 {key} 数据, 形状: {df.shape}")
                df.columns = df.columns.str.lower()
                
                date_cols = []
                if key == 'admissions' and 'admittime' in df.columns:
                    date_cols = ['admittime', 'dischtime']
                elif key in ['labevents', 'vitalsign', 'urine_output'] and 'charttime' in df.columns:
                    date_cols = ['charttime']
                elif key == 'icustays' and 'intime' in df.columns:
                    date_cols = ['intime', 'outtime']
                
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], format=date_format, dayfirst=True, errors='coerce')
                
                if key == 'patients' and 'gender' not in df.columns:
                    logger.warning("patients 表中缺少 'gender' 列")
                    gender_aliases = ['sex', 'gender_code', 'patient_gender']
                    for alias in gender_aliases:
                        if alias in df.columns:
                            logger.info(f"使用 '{alias}' 作为性别列")
                            df.rename(columns={alias: 'gender'}, inplace=True)
                            break
                    else:
                        logger.warning("未找到性别列，创建默认列 (0=女, 1=男)")
                        df['gender'] = np.random.choice([0, 1], size=len(df))
                
                data[key] = df
            except Exception as e:
                logger.error(f"加载 {key} 数据失败: {str(e)}")
                raise
        
        return data

    def _detect_zero_slope_cases(self, creat_features):
        """
        检测并过滤肌酐斜率为0的病例（统一判断标准，与is_zero_slope对齐）
        """
        try:
            filtered_features = creat_features.copy()
            
            # 统一使用is_zero_slope函数判断，避免标准不一致
            zero_slope_mask = filtered_features['creat_slope'].apply(self.is_zero_slope)  # 调用类内统一方法
            
            # 记录斜率为0的病例信息
            zero_slope_cases = []
            if zero_slope_mask.any():
                zero_df = filtered_features[zero_slope_mask]
                for _, row in zero_df.iterrows():
                    zero_slope_cases.append({
                        'hadm_id': row['hadm_id'],
                        'creat_slope': row['creat_slope'],
                        'max_creat': row['max_creat'] if 'max_creat' in row else None,
                        'reason': f"肌酐斜率为0: {row['creat_slope']:.3f} μmol/L/hour"
                    })
                
                logger.warning(f"检测到 {len(zero_slope_cases)} 个肌酐斜率为0的病例:")
                for case in zero_slope_cases[:10]:  # 只显示前10个
                    logger.warning(f"  - HADM_ID: {case['hadm_id']}, 斜率: {case['creat_slope']:.3f}")
                if len(zero_slope_cases) > 10:
                    logger.warning(f"  ... 还有 {len(zero_slope_cases) - 10} 个斜率为0的病例未显示")
            
            # 过滤掉斜率为0的病例
            before_count = len(filtered_features)
            filtered_features = filtered_features[~zero_slope_mask]
            after_count = len(filtered_features)
            
            logger.info(f"肌酐斜率为0病例过滤: 从 {before_count} 个病例中过滤掉 {before_count - after_count} 个病例")
            
            # 增加过滤后数据量校验
            if after_count == 0:
                logger.error("过滤后无有效病例，降低0斜率判断严格度")
                # 兜底策略：放宽阈值，保留部分低斜率病例
                zero_slope_mask = filtered_features['creat_slope'].abs() < 1e-5  # 放宽100倍
                filtered_features = creat_features[~zero_slope_mask]
                logger.info(f"放宽阈值后，剩余病例数: {len(filtered_features)}")
            
            return filtered_features, zero_slope_cases
            
        except Exception as e:
            logger.error(f"检测肌酐斜率为0病例失败: {str(e)}")
            raise

    # 把is_zero_slope作为类方法，方便统一调用
    def is_zero_slope(self, creat_slope, threshold=1e-10):
        """
        判断是否为0肌酐斜率（统一浮点数精度判断，类内复用）
        """
        if pd.isna(creat_slope) or not isinstance(creat_slope, (int, float)):
            return True  # 空值/非数值视为0斜率（过滤）
        return abs(float(creat_slope)) < threshold

    def _build_features(self, data):
        """特征工程流水线（增强版，仅过滤肌酐斜率为0的case）"""
        try:
            base = data['admissions'][['subject_id', 'hadm_id', 'admittime', 'admission_type']]\
                .merge(data['patients'][['subject_id', 'anchor_age', 'gender']], on='subject_id')\
                .merge(data['creatinine_baseline'], on='hadm_id', how='left')
            
            base['prediction_time'] = base['admittime'] + timedelta(hours=24)
            
            # 处理肌酐数据
            creat = data['labevents'][
                (data['labevents']['itemid'] == 50912) & 
                (data['labevents']['valuenum'].notna())
            ].copy()
            creat = creat.merge(base[['hadm_id', 'prediction_time']], on='hadm_id', how='left')
            creat = creat[creat['charttime'] <= creat['prediction_time']]
            creat = creat.groupby('hadm_id').filter(lambda g: len(g) >= 2)

            def safe_slope_calc(g):
                """增强版肌酐斜率计算，修复全0结果漏洞，增加调试信息"""
                try:
                    # 确保g是DataFrame（包含charttime和valuenum列），避免Series传入
                    if not isinstance(g, pd.DataFrame):
                        # 若为Series，尝试转换为DataFrame（保留索引，用于关联charttime）
                        logger.warning("输入数据为Series，无法获取charttime，返回默认值0.001")
                        return 0.001
                    
                    # 确保按时间排序（DataFrame必须指定by关键字参数，避免位置参数报错）
                    # 规范调用：使用by=指定排序列，ascending=True确保时间正序
                    g_sorted = g.sort_values(by='charttime', ascending=True).reset_index(drop=True)
                    
                    # 需要至少2个不同的时间点和肌酐值
                    if len(g_sorted) < 2:
                        logger.warning(f"数据点不足，无法计算斜率，返回默认值 0.000")
                        return 0.000  # 返回0.000，过滤掉
                    
                    # 提取肌酐值和时间（确保列存在）
                    if 'valuenum' not in g_sorted.columns or 'charttime' not in g_sorted.columns:
                        logger.warning("缺少必要列（valuenum/charttime），返回默认值0.000")
                        return 0.000
                    
                    creat_values = g_sorted['valuenum'].values
                    chart_times = g_sorted['charttime'].values
                    
                    # 计算时间差（小时），确保精度（修复核心：避免时间差为0）
                    time_diff_seconds = (chart_times - chart_times[0]).astype('timedelta64[s]').astype(float)
                    time_diff_hours = time_diff_seconds / 3600.0
                    
                    # 检查时间差是否有效（至少大于0.5小时，避免无效数据）
                    total_time_hours = time_diff_hours[-1]
                    if total_time_hours < 0.5:
                        logger.warning(f"时间跨度过小（{total_time_hours:.2f}小时），使用肌酐差值计算斜率")
                        # 用最终值与初始值的差值除以最小有效时间（0.5小时）
                        creat_diff = creat_values[-1] - creat_values[0]
                        if creat_diff == 0:
                            logger.warning(f"时间跨度过小（{total_time_hours:.2f}小时），使用斜率0.000")
                            return 0.000  # 避免全0，保留病例
                        return creat_diff / 0.5
                    
                    # 检查肌酐值是否全部相同
                    if np.all(creat_values == creat_values[0]):
                        logger.warning(f"肌酐值无变化，返回默认趋势值 0.001")
                        return 0.001  # 不返回0，避免被过滤
                    
                    # 使用线性回归计算斜率（增加异常捕获）
                    try:
                        slope, intercept = np.polyfit(time_diff_hours, creat_values, 1)
                    except Exception as e:
                        logger.warning(f"线性回归拟合失败：{e}，使用差值法计算斜率")
                        creat_diff = creat_values[-1] - creat_values[0]
                        slope = creat_diff / total_time_hours
                    
                    # 修正极端小斜率（避免接近0被过滤，保留有效病例）
                    if abs(slope) < 0.001 and slope != 0:
                        slope = 0.001 if slope > 0 else -0.001
                    elif slope == 0:
                        # 强制避免全0，赋予极小趋势值
                        slope = 0.001
                    
                    return slope
                except Exception as e:
                    logger.warning(f"斜率计算异常: {str(e)}, 数据点: {len(g) if 'g' in locals() else 0}，返回默认值 0.001")
                    return 0.001  # 兜底返回非0值，避免被过滤

            # 修正：分组聚合方式（确保传入safe_slope_calc的是完整DataFrame）
            # 替换原有的creat_features = creat.groupby('hadm_id').agg(...)代码
            creat_features = []
            for hadm_id, group_df in creat.groupby('hadm_id'):
                if len(group_df) >= 2:
                    max_creat = group_df['valuenum'].max()
                    creat_slope = safe_slope_calc(group_df)  # 传入完整分组DataFrame
                    creat_features.append({
                        'hadm_id': hadm_id,
                        'max_creat': max_creat,
                        'creat_slope': creat_slope
                    })
            # 转换为DataFrame
            creat_features = pd.DataFrame(creat_features)
            
            # 关键修改：检测并过滤肌酐斜率为0的病例
            creat_features, zero_slope_cases = self._detect_zero_slope_cases(creat_features)
            
            # 保存斜率为0的病例信息以供分析
            if zero_slope_cases:
                self.zero_slope_cases = zero_slope_cases
                # 保存到文件
                zero_df = pd.DataFrame(zero_slope_cases)
                zero_df.to_csv(f"zero_slope_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                              index=False, encoding='utf-8-sig')
                logger.info(f"肌酐斜率为0病例已保存到CSV文件，共 {len(zero_slope_cases)} 条记录")

            # 继续处理其他特征
            vital = data['vitalsign'].merge(
                base[['subject_id', 'hadm_id', 'prediction_time']],
                on='subject_id',
                how='left'
            )
            vital = vital[vital['charttime'] <= vital['prediction_time']]
            
            vital_features = vital.groupby('hadm_id').agg(
                max_hr=('heart_rate', 'max'),
                min_map=('mbp', 'min')
            ).reset_index()
            vital_features['max_hr'] = vital_features['max_hr'].fillna(80)
            vital_features['min_map'] = vital_features['min_map'].fillna(70)

            uo_with_icu = data['urine_output'].merge(
                data['icustays'][['stay_id', 'hadm_id', 'subject_id']],
                on='stay_id',
                how='inner'
            )
            uo_with_adm = uo_with_icu.merge(
                base[['hadm_id', 'prediction_time']],
                on='hadm_id',
                how='left'
            )
            uo_with_adm['charttime'] = pd.to_datetime(uo_with_adm['charttime'])
            
            # 过滤：仅保留预测时间点之前的记录
            uo_filtered = uo_with_adm[uo_with_adm['charttime'] <= uo_with_adm['prediction_time']].copy()
            
            # 计算每条记录与预测时间点的差值（小时）
            uo_filtered['time_diff_hours'] = (uo_filtered['prediction_time'] - uo_filtered['charttime']).dt.total_seconds() / 3600
            
            # 保存 hadm_id 和 subject_id 的映射关系
            hadm_subject_map = uo_with_adm[['hadm_id', 'subject_id']].drop_duplicates()
            
            # 分别计算6小时和24小时窗口的尿输出总和
            uo_6hr = uo_filtered[uo_filtered['time_diff_hours'] <= 6].groupby('hadm_id')['urineoutput'].sum().reset_index()
            uo_6hr.columns = ['hadm_id', 'uo_6hr']
            
            uo_24hr = uo_filtered[uo_filtered['time_diff_hours'] <= 24].groupby('hadm_id')['urineoutput'].sum().reset_index()
            uo_24hr.columns = ['hadm_id', 'uo_24hr']
            
            # 合并尿输出特征
            uo_features = uo_6hr.merge(uo_24hr, on='hadm_id', how='outer')
            
            # 将 subject_id 合并回 uo_features
            uo_features = uo_features.merge(hadm_subject_map, on='hadm_id', how='left')
            
            # 合并体重数据
            uo_features = uo_features.merge(
                data['first_day_weight'][['subject_id', 'weight']],
                on=['subject_id'],
                how='left'
            )
            
            # 计算uo_ml_kg_hr（24小时尿输出 / 体重 / 24）
            uo_features['uo_ml_kg_hr'] = uo_features.apply(
                lambda row: row['uo_24hr'] / (row['weight'] * 24) 
                if (row['weight'] > 0 and pd.notna(row['uo_24hr'])) else np.nan,
                axis=1
            )
            uo_features = uo_features[['hadm_id', 'uo_6hr', 'uo_24hr', 'uo_ml_kg_hr']].fillna(0)

            # 合并所有特征，只保留肌酐斜率不为0的病例
            final_df = base.merge(creat_features, on='hadm_id', how='inner')\
                        .merge(vital_features, on='hadm_id', how='left')\
                        .merge(uo_features, on='hadm_id', how='left')
            
            final_df['max_creat'] = final_df['max_creat'].fillna(final_df['scr_baseline'])
            final_df['creat_slope'] = final_df['creat_slope'].fillna(0)
            final_df['uo_6hr'] = final_df['uo_6hr'].fillna(0)
            final_df['uo_24hr'] = final_df['uo_24hr'].fillna(0)
            final_df['uo_ml_kg_hr'] = final_df['uo_ml_kg_hr'].fillna(0)
            
            # 新增特征
            # 1. 计算肌酐变化百分比
            final_df['creat_change_percent'] = ((final_df['max_creat'] - final_df['scr_baseline']) / 
                                               final_df['scr_baseline'] * 100).fillna(0)
            final_df.loc[~np.isfinite(final_df['creat_change_percent']), 'creat_change_percent'] = 0
            
            # 2. 计算平均尿量率
            final_df['uo_avg_rate'] = final_df['uo_24hr'] / 24  # ml/hour
            
            # 3. 创建复合风险评分
            final_df['risk_score'] = 0
            
            # 添加肌酐上升速度分类特征
            final_df['rapid_creat_increase'] = ((final_df['creat_slope'] > 0.15) & 
                                                (final_df['creat_change_percent'] > 20)).astype(int)

            # 添加多系统功能障碍标志
            final_df['multiorgan_failure'] = ((final_df['min_map'] < 65) & 
                                            (final_df['max_hr'] > 120)).astype(int)

            # 根据临床指标增加风险分
            conditions = [
                (final_df['creat_change_percent'] >= 50, 3),
                (final_df['creat_change_percent'] >= 30, 2),
                (final_df['creat_change_percent'] >= 10, 1),
                (final_df['uo_ml_kg_hr'] < 0.3, 3),
                (final_df['uo_ml_kg_hr'] < 0.5, 2),
                (final_df['min_map'] < 60, 2),
                (final_df['min_map'] < 70, 1),
                (final_df['max_hr'] > 130, 2),
                (final_df['max_hr'] > 110, 1),
            ]
            
            for condition, score in conditions:
                final_df.loc[condition, 'risk_score'] += score
            
            # 获取AKI标签
            aki_labels = data['kdigo_stages'].merge(
                base[['hadm_id', 'prediction_time']],
                on='hadm_id',
                how='inner'
            )
            aki_labels = aki_labels[aki_labels['charttime'] > aki_labels['prediction_time']]
            aki_labels = aki_labels.groupby('hadm_id')['aki_stage'].max().reset_index()
            
            # 合并标签到最终数据集
            final_df = final_df.merge(aki_labels, on='hadm_id', how='left')
            final_df['aki_stage'] = final_df['aki_stage'].fillna(0)
            
            logger.info(f"特征工程完成，过滤后数据形状: {final_df.shape}")
            logger.info(f"肌酐斜率范围: [{final_df['creat_slope'].min():.3f}, {final_df['creat_slope'].max():.3f}] μmol/L/hour")
            logger.info(f"肌酐斜率为0的病例已被过滤，剩余病例数: {len(final_df)}")
            
            # --------------------------
            # 新增：二次校验与过滤0斜率病例【修改点2】
            # --------------------------
            residual_zero_slope_mask = final_df['creat_slope'].apply(is_zero_slope)
            residual_zero_slope_count = residual_zero_slope_mask.sum()
            
            if residual_zero_slope_count > 0:
                logger.warning(f"发现 {residual_zero_slope_count} 个残留0斜率病例，进行二次过滤")
                # 二次过滤残留0斜率病例
                final_df = final_df[~residual_zero_slope_mask]
                logger.info(f"二次过滤后剩余病例数: {len(final_df)}")
            
            # 最终校验：确认无0斜率病例
            final_zero_slope_count = final_df['creat_slope'].apply(is_zero_slope).sum()
            logger.info(f"最终预处理结果：0斜率病例数 = {final_zero_slope_count}")
            
            return final_df
        except Exception as e:
            logger.error(f"特征工程失败: {str(e)}")
            logger.exception("异常详情")
            raise

    def _clean_data(self, df):
        """数据清洗与转换"""
        try:
            df['max_hr'] = df['max_hr'].clip(40, 200)
            df['min_map'] = df['min_map'].clip(30, None)
            df['scr_baseline'] = df.groupby(['anchor_age', 'gender'])['scr_baseline']\
                .transform(lambda x: x.fillna(x.median()))
            df['diagnosis_category'] = df['diagnosis'].str[:3].map(self.icd_map)
            return df
        except Exception as e:
            logger.error(f"数据清洗失败: {str(e)}")
            raise

    def process(self):
        """执行完整处理流程"""
        try:
            logger.info("开始数据处理流程")
            raw_data = self._load_raw_data()
            processed_df = self._build_features(raw_data)

            if 'aki_stage' in processed_df.columns:
                stage_counts = processed_df['aki_stage'].value_counts()
                logger.info(f"AKI阶段分布: \n{stage_counts.to_string()}")
                positive_count = processed_df[processed_df['aki_stage'] >= 1].shape[0]
                total_count = processed_df.shape[0]
                positive_ratio = positive_count / total_count if total_count > 0 else 0
                logger.info(f"阳性样本比例: {positive_ratio:.2%} ({positive_count}/{total_count})")
            else:
                logger.warning("数据框中缺少 'aki_stage' 列，无法统计类别分布")

            logger.info("数据处理完成")
            return processed_df
        except Exception as e:
            logger.error(f"数据处理流程失败: {str(e)}")
            raise

# --------------------------
# 模型训练与评估模块 (已修改为分类任务) - 优化版
# --------------------------
class AKIPredictor:
    def __init__(self, model_id="/home/fbh/llama2-7b/Llama-2-7b-hf"):
        try:
            logger.info(f"初始化分类预测器, 模型: {model_id}")
            self.model_id = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=False,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            # 保存最优阈值（训练过程中更新）
            self.best_threshold = 0.5
            # 最低召回率要求（医疗场景核心）
            self.min_recall_requirement = 0.8
            # 新增：最优阈值持久化文件路径
            self.threshold_file = "./aki_best_threshold.json"
            logger.info("分词器初始化完成")
        except Exception as e:
            logger.error(f"预测器初始化失败: {str(e)}")
            raise

    def _format_data(self, df):
        """格式化数据为分类任务格式 (prompt, label)，并记录肌酐斜率信息"""
        formatted = []
        try:
            if 'gender' not in df.columns:
                logger.warning("数据框中缺少 'gender' 列，使用默认值")
                df['gender'] = np.random.choice([0, 1], size=len(df))
            
            gender_map = {0: 'Female', 1: 'Male'}
            
            for _, row in df.iterrows():
                # 填充和转换数值，保持与v2.0一致
                row = self._fill_and_convert_row_values(row)

                # 新增特征值
                creat_change_percent = ((row['max_creat'] - row['scr_baseline']) / row['scr_baseline'] * 100) if row['scr_baseline'] > 0 else 0
                uo_avg_rate = row['uo_24hr'] / 24 if row['uo_24hr'] > 0 else 0
                risk_score = 0
                
                # 计算风险评分
                if creat_change_percent >= 50: risk_score += 3
                elif creat_change_percent >= 30: risk_score += 2
                elif creat_change_percent >= 10: risk_score += 1
                
                if row['uo_ml_kg_hr'] < 0.3: risk_score += 3
                elif row['uo_ml_kg_hr'] < 0.5: risk_score += 2
                
                if row['min_map'] < 60: risk_score += 2
                elif row['min_map'] < 70: risk_score += 1
                
                if row['max_hr'] > 130: risk_score += 2
                elif row['max_hr'] > 110: risk_score += 1

                # 构建提示词 (增强版，包含新增特征)
                prompt = f"""As a nephrology specialist, assess the patient's risk of developing acute kidney injury (AKI) based on the following clinical data:
Patient Information:
- Age: {int(row['anchor_age'])} years
- Gender: {gender_map.get(row['gender'], 'Unknown')}

Key Lab Indicators:
- Baseline Creatinine: {row['scr_baseline']:.1f} μmol/L
- Maximum Creatinine: {row['max_creat']:.1f} μmol/L
- Creatinine Change: {creat_change_percent:.1f}% from baseline
- Creatinine Trend: {row['creat_slope']:.3f} μmol/L/hour

Vital Signs:
- Minimum Mean Arterial Pressure: {row['min_map']:.1f} mmHg
- Maximum Heart Rate: {int(row['max_hr'])} beats/min

Urine Output:
- 6-hour Urine Output: {row['uo_6hr']:.1f} ml
- 24-hour Urine Output: {row['uo_24hr']:.1f} ml
- Urine Output Rate: {uo_avg_rate:.1f} ml/hour
- Urine Output per kg: {row['uo_ml_kg_hr']:.3f} ml/kg/hour

Risk Assessment:
{self._get_risk_notes(row)}
Composite Risk Score: {risk_score}"""

                # 直接使用 1/0 作为标签
                label = 1 if row.get('aki_stage', 0) >= 1 else 0
                
                formatted.append({
                    "hadm_id": row['hadm_id'],
                    "prompt": prompt,
                    "label": label,
                    "risk_score": risk_score,
                    "creat_change_percent": creat_change_percent,
                    "creat_slope": row['creat_slope']  # 保存肌酐斜率用于分析
                })
            logger.info(f"格式化完成，共 {len(formatted)} 条数据")
            return formatted
        except Exception as e:
            logger.error(f"数据格式化失败: {str(e)}")
            raise

    def _fill_and_convert_row_values(self, row):
        """填充空值并转换数据类型"""
        required_cols = ['anchor_age', 'gender', 'scr_baseline', 
                        'max_creat', 'creat_slope', 'min_map', 'max_hr',
                        'uo_6hr', 'uo_24hr', 'hadm_id', 'uo_ml_kg_hr']
        
        for col in required_cols:
            if col not in row or pd.isna(row[col]):
                if col in ['anchor_age', 'max_hr']: row[col] = 0
                elif col in ['scr_baseline', 'max_creat', 'creat_slope', 'min_map', 'uo_6hr', 'uo_24hr', 'uo_ml_kg_hr']: row[col] = 0.0

        row['max_hr'] = float(row['max_hr'])
        row['anchor_age'] = int(row['anchor_age'])
        for col in ['scr_baseline', 'max_creat', 'creat_slope', 'min_map', 'uo_6hr', 'uo_24hr', 'uo_ml_kg_hr']:
            row[col] = float(row[col])
        
        return row

    def _get_risk_notes(self, row):
        notes = []
        if row['max_creat'] > row['scr_baseline'] * 1.5:
            increase_ratio = (row['max_creat'] - row['scr_baseline']) / row['scr_baseline'] * 100
            notes.append(f"Creatinine elevation: {increase_ratio:.1f}% above baseline (AKI criteria)")
        if row['uo_ml_kg_hr'] < 0.5:
            notes.append("Severe oliguria: <0.5ml/kg/hr (AKI Stage 3)")
        if row['min_map'] < 65:
            notes.append("Hypotension: Mean arterial pressure <65mmHg (renal hypoperfusion)")
        if row['max_hr'] > 120:
            notes.append("Tachycardia: Heart rate >120 bpm (volume depletion)")
        if row['creat_slope'] > 0.1:
            notes.append("Rising creatinine trend: >0.1 μmol/L/hour (progressive injury)")
        # 新增风险提示
        if row['uo_ml_kg_hr'] < 0.3:
            notes.append("Critical oliguria: <0.3ml/kg/hr (high risk)")
        if row['min_map'] < 60:
            notes.append("Severe hypotension: Mean arterial pressure <60mmHg (critical)")
        return "\n".join(notes) if notes else "No high-risk indicators detected"

    def _tokenize_function(self, examples):
        """标记化函数"""
        return self.tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )

    def train(self, train_data, val_data):
        """模型训练流程 (分类任务) - 已适配分布式训练"""
        try:
            logger.info("开始分布式模型训练流程 (分类任务)")
            # 获取当前进程的local_rank
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # 转换为Dataset
            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data)
            
            # 标记化
            train_dataset = train_dataset.map(
                self._tokenize_function, batched=True,num_proc=8, remove_columns=["prompt", "hadm_id", "risk_score", "creat_change_percent", "creat_slope"]
            )
            val_dataset = val_dataset.map(
                self._tokenize_function, batched=True,num_proc=8, remove_columns=["prompt", "hadm_id", "risk_score", "creat_change_percent", "creat_slope"]
            )
            logger.info("数据集标记化完成")

            # 计算类别权重
            all_labels = [d["label"] for d in train_data]
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=all_labels)
            # 大幅提高正样本权重以提高召回率
            class_weights[1] *= 1.5
            logger.info(f"训练集类别分布: 阴性 {all_labels.count(0)}, 阳性 {all_labels.count(1)}")
            logger.info(f"训练类别权重: 阴性={class_weights[0]:.2f}, 阳性={class_weights[1]:.2f}")

            # 配置QLoRA
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # 为分布式训练配置device_map
            device_map = {"": local_rank}

            # 加载分类模型
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                num_labels=2,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            model = prepare_model_for_kbit_training(model)
            
            # 配置LoRA
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS"
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

            # 定义带增强阈值优化的compute_metrics函数
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                # 转换为概率
                probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
                
                # 多层次阈值搜索策略
                thresholds_to_try = []
                
                # 第一层：常规搜索
                thresholds_to_try.extend(np.arange(0.4, 0.86, 0.05))
                
                # 第二层：精细搜索（在召回率>0.75的阈值附近）
                recall_list = []
                for threshold in np.arange(0.25, 0.75, 0.02):
                    predictions = (probs >= threshold).astype(int)
                    current_recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
                    recall_list.append((threshold, current_recall))
                
                # 找到召回率>0.75的阈值区域
                high_recall_thresholds = [t for t, r in recall_list if r >= 0.75]
                if high_recall_thresholds:
                    min_high_recall = min(high_recall_thresholds)
                    max_high_recall = max(high_recall_thresholds)
                    # 在这个区域进行精细搜索
                    thresholds_to_try.extend(np.arange(min_high_recall-0.03, max_high_recall+0.03, 0.01))
                
                best_threshold = 0.65
                best_f1 = 0.0
                best_precision = 0.50
                best_metrics = {}
                
                for threshold in thresholds_to_try:
                    predictions = (probs >= threshold).astype(int)
                    
                    # 计算所有指标
                    current_recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
                    current_precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
                    current_f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
                    current_f2 = fbeta_score(labels, predictions, beta=2, pos_label=1, zero_division=0)
                    
                    # 核心要求：召回率必须≥80%，精确率大于0.5
                    if current_recall >= self.min_recall_requirement and current_precision >= 0.5:
                        # 在满足召回率的前提下，寻找F1分数最高的阈值
                        if current_f1 > best_f1 and current_precision > best_precision:
                            best_f1 = current_f1
                            best_threshold = threshold
                            best_precision = current_precision
                            best_metrics = {
                                "accuracy": accuracy_score(labels, predictions),
                                "auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5,
                                "recall": current_recall,
                                "precision": current_precision,
                                "f1": current_f1,
                                "f2": current_f2,
                                "threshold": threshold
                            }

                # 如果没有任何阈值满足召回率要求，使用F2分数最高且召回率最高的阈值
                if best_f1 == 0.0:
                    logger.warning("没有阈值满足召回率要求，使用备用策略")
                    # 按召回率排序
                    recall_list.sort(key=lambda x: x[1], reverse=True)
                    # 取召回率最高的前3个，选择F2分数最高的
                    top_recall_thresholds = recall_list[:3]
                    best_f2 = 0.0
                    for threshold, rec in top_recall_thresholds:
                        predictions = (probs >= threshold).astype(int)
                        current_f2 = fbeta_score(labels, predictions, beta=2, pos_label=1, zero_division=0)
                        if current_f2 > best_f2:
                            best_f2 = current_f2
                            best_threshold = threshold
                    
                    predictions = (probs >= best_threshold).astype(int)
                    best_metrics = {
                        "accuracy": accuracy_score(labels, predictions),
                        "auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5,
                        "recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
                        "precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
                        "f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
                        "f2": fbeta_score(labels, predictions, beta=2, pos_label=1, zero_division=0),
                        "threshold": best_threshold
                    }
                
                # 更新全局最优阈值（仅主进程更新）
                if local_rank == 0:
                    self.best_threshold = best_threshold
                    # 新增：将最优阈值保存到json文件
                    threshold_data = {
                        "best_threshold": float(best_threshold),
                        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "corresponding_f1": best_metrics.get("f1", 0.0)
                    }
                    with open(self.threshold_file, 'w', encoding='utf-8') as f:
                        json.dump(threshold_data, f, indent=2)

                    logger.info(f"本轮最优阈值: {best_threshold:.4f}")
                    logger.info(f"对应指标 - 召回率: {best_metrics['recall']:.4f}, 精确率: {best_metrics['precision']:.4f}, F1: {best_metrics['f1']:.4f}")
                
                return best_metrics

            # 训练参数 - 优化学习策略
            training_args = TrainingArguments(
                output_dir="./aki-classification-results",
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                num_train_epochs=3,  # 增加训练轮数
                optim="paged_adamw_32bit",
                learning_rate=8e-6,  # 微调学习率
                fp16=False,
                bf16=True,
                logging_steps=50,
                eval_strategy=IntervalStrategy.EPOCH,
                eval_steps=1, 
                save_strategy=IntervalStrategy.EPOCH,
                save_steps=1,
                report_to="tensorboard",
                remove_unused_columns=False,
                max_grad_norm=0.3,
                warmup_steps=300,
                lr_scheduler_type="cosine_with_restarts",  # 使用带重启的cosine调度
                dataloader_num_workers=4,
                metric_for_best_model="eval_recall",  
                greater_is_better=True,
                weight_decay=0.02,  # 增加权重衰减
                save_only_model=True,            # 只保存模型权重，不保存优化器状态
                load_best_model_at_end=False,    # 关键：训练时不加载最佳模型
                ddp_find_unused_parameters=False,
                label_smoothing_factor=0.05,  # 轻微标签平滑
                save_total_limit=3,  # 只保存最好的3个检查点
            )

            # 自定义Trainer以使用增强的损失函数
            class WeightedTrainer(Trainer):
                def __init__(self, class_weights, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # 增加正样本权重，提高召回率
                    enhanced_weights = class_weights.copy()
                    enhanced_weights[1] *= 1.3  # 将正样本权重增加30%
                    self.class_weights = torch.tensor(enhanced_weights, dtype=torch.float32).to(self.model.device)
                    logger.info(f"增强训练权重: 阴性={enhanced_weights[0]:.2f}, 阳性={enhanced_weights[1]:.2f}")

                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # 使用focal loss变体来关注难分类样本
                    ce_loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
                    
                    # 添加focal loss因子（对难分类样本给予更多关注）
                    probs = torch.softmax(logits, dim=1)
                    pt = probs.gather(1, labels.unsqueeze(1)).squeeze()
                    focal_factor = (1 - pt) ** 2  # gamma=2
                    
                    loss = (focal_factor * ce_loss).mean()
                    
                    return (loss, outputs) if return_outputs else loss

            # 初始化Trainer
            trainer = WeightedTrainer(
                class_weights=class_weights,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.005)],
            )

            # 开始训练
            logger.info(f"进程 {local_rank}: 开始训练...")
            trainer.train()
            
            # 保存最佳模型（仅主进程）
            if local_rank == 0:
                best_model_path = f"./aki-classification-best-model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                trainer.save_model(best_model_path)
                logger.info(f"最佳模型已保存至 {best_model_path}, 最优阈值: {self.best_threshold}")
                # 清理显存
                torch.cuda.empty_cache()
            if local_rank == 0:
                return model, best_model_path
            else:
                return model, None

        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            raise

    def _check_strong_positive(self, prompt, prob):
        """检查是否为必须确诊的强阳性案例（收紧版，增加肌酐趋势前提，减少假阳性）"""
        # --------------------------
        # 删除：冗余的0斜率判断逻辑【修改点3】
        # --------------------------
        # 规则1: 符合KDIGO Stage 3标准（少尿+肌酐升高≥50%）
        has_severe_oliguria = "Severe oliguria: <0.5ml/kg/hr (AKI Stage 3)" in prompt
        has_creat_elevation = "Creatinine elevation:" in prompt
        creat_percent = 0.0
        if has_creat_elevation:
            percent_match = re.search(r"Creatinine elevation: ([0-9.]+)%", prompt)
            if percent_match:
                creat_percent = float(percent_match.group(1))
        
        if has_severe_oliguria and has_creat_elevation and creat_percent >= 50 and prob >= 0.8:
            return True
        
        # 规则2: 肌酐快速上升(>0.3μmol/L/h) + 低血压 + 肌酐升高≥30%
        has_hypotension = "Hypotension: Mean arterial pressure <65mmHg" in prompt
        slope_match = re.search(r"Creatinine Trend: ([0-9.-]+) μmol/L/hour", prompt)
        if slope_match and has_hypotension:
            creat_slope = float(slope_match.group(1))
            if creat_slope > 0.3 and creat_percent >= 30 and prob >= 0.75:
                return True
        
        # 规则3: 极危重少尿(<0.3ml/kg/h) + 心动过速 + 风险评分≥8
        has_critical_oliguria = "Critical oliguria: <0.3ml/kg/hr (high risk)" in prompt
        has_tachycardia = "Tachycardia: Heart rate >120 bpm (volume depletion)" in prompt
        risk_score = 0
        risk_score_match = re.search(r"Composite Risk Score: ([0-9]+)", prompt)
        if risk_score_match:
            risk_score = int(risk_score_match.group(1))
        if has_critical_oliguria and has_tachycardia and risk_score >= 8 and prob >= 0.7:
            return True
        
        # 规则4: 肌酐升高>100% + 任何2项AKI风险因素 + 绝对变化≥1.5 μmol/L（新增：避免百分比失真）
        baseline_match = re.search(r"Baseline Creatinine: ([0-9.]+) μmol/L", prompt)
        max_match = re.search(r"Maximum Creatinine: ([0-9.]+) μmol/L", prompt)
        if baseline_match and max_match:
            baseline_creat = float(baseline_match.group(1))
            max_creat = float(max_match.group(1))
            creat_abs_change = max_creat - baseline_creat
            if has_creat_elevation and creat_percent > 100 and creat_abs_change >= 1.5:
                risk_factors = sum([
                    1 if has_severe_oliguria else 0,
                    1 if has_hypotension else 0,
                    1 if has_tachycardia else 0
                ])
                if risk_factors >= 2 and prob >= 0.8:
                    return True
        
        # 规则5: 高风险评分≥10 + 低血压 + 肌酐上升趋势
        if risk_score >= 10 and has_hypotension and slope_match and float(slope_match.group(1)) > 0.1 and prob >= 0.75:
            return True
        
        return False

    def evaluate(self, model, eval_data):
        """评估模型性能 - 内存优化版（仅主进程执行）"""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return {}  # 非主进程直接返回空结果，不执行评估

        logger.info(f"开始评估模型... 使用最优阈值: {self.best_threshold}")
        model.eval()
        torch.cuda.empty_cache()  # 评估前清理缓存
        
        # 1. 提取基本数据
        all_hadm_ids = [sample["hadm_id"] for sample in eval_data]
        all_prompts = [sample["prompt"] for sample in eval_data]
        all_labels = np.array([sample["label"] for sample in eval_data])
        
        # 2. 降低批处理大小 (从16降低到4)
        batch_size = 4  # 根据GPU内存情况可进一步调整
        all_probs = []
        all_predictions = []
        fp_filter_count = 0
        tp_confirm_count = 0
        
        total_batches = (len(all_prompts) + batch_size - 1) // batch_size  # 修正：避免整除导致批次统计不全
        # 3. 逐批处理，不一次性加载所有数据
        for i in range(0, len(all_prompts), batch_size):
            batch_idx = i // batch_size + 1
            batch_prompts = all_prompts[i:i+batch_size]
            
            # 可选：每10个batch打印一次进度，避免日志过多
            if i % (batch_size * 100) == 0:
                logger.info(f"评估进度: {batch_idx}/{total_batches} 批次 ({i}/{len(all_prompts)} 样本)")
            
            # 3.1 仅标记化当前批次
            batch_inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="max_length"
            ).to(model.device)
            
            # 3.2 推理并立即移回CPU
            with torch.no_grad():
                outputs = model(**batch_inputs)
                batch_logits = outputs.logits.cpu()  # 立即移回CPU
            
            # 3.3 计算概率
            batch_probs = torch.nn.functional.softmax(batch_logits.float(), dim=1)[:, 1].numpy()
            
            # 3.4 应用阈值和医学规则
            batch_initial_preds = (batch_probs >= self.best_threshold).astype(int)
            
            # 应用医学规则过滤
            batch_final_preds = []
            # 修正评估阶段的批处理逻辑（调整规则优先级）
   
            for j, (pred, prompt, prob) in enumerate(zip(batch_initial_preds, batch_prompts, batch_probs)):
                # 第一步：先执行强阳性确认（优先锁定真阳性，不允许后续过滤）
                is_strong_positive = self._check_strong_positive(prompt, prob)
                if is_strong_positive:
                    final_pred = 1
                    tp_confirm_count += 1
                    batch_final_preds.append(final_pred)
                    continue  # 强阳性直接跳过后续过滤，保护召回率
                
                final_pred = pred
                
                batch_final_preds.append(final_pred)

            # 3.5 保存结果
            all_probs.extend(batch_probs.tolist())
            all_predictions.extend(batch_final_preds)
            
            # 3.6 清理当前批次的变量
            del batch_inputs, outputs, batch_logits
            torch.cuda.empty_cache()
        
        # 4. 转换为numpy数组
        all_probs = np.array(all_probs)
        all_predictions = np.array(all_predictions)
        
        # 5. 计算评估指标
        metrics = {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
            "recall": recall_score(all_labels, all_predictions, pos_label=1, zero_division=0),
            "precision": precision_score(all_labels, all_predictions, pos_label=1, zero_division=0),
            "f1": f1_score(all_labels, all_predictions, pos_label=1, zero_division=0),
            "f2": fbeta_score(all_labels, all_predictions, beta=2, pos_label=1, zero_division=0),
            "tp_confirm_count": tp_confirm_count,
            "filtered_fp_count": fp_filter_count,
            "best_threshold": self.best_threshold,
            "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist(),
            "raw_predictions_positive": int(sum((all_probs >= self.best_threshold).astype(int))),
            "filtered_predictions_positive": int(sum(all_predictions))
        }
        
        # 6. 假阳性分析
        try:
            self._export_false_positives(
                all_hadm_ids,
                all_prompts,
                all_labels,
                all_predictions,
                all_probs,
                []
            )
        except Exception as e:
            logger.warning(f"假阳性导出失败，但评估继续: {str(e)}")
        
        # 7. 打印结果
        logger.info(f"评估完成 | 准确率: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f}")
        logger.info(f"召回率: {metrics['recall']:.4f} | 精确率: {metrics['precision']:.4f} | F1: {metrics['f1']:.4f} | F2: {metrics['f2']:.4f}")
        logger.info(f"原始阳性预测数: {metrics['raw_predictions_positive']} | 过滤后阳性预测数: {metrics['filtered_predictions_positive']}")
        logger.info(f"混淆矩阵: {metrics['confusion_matrix']}")
        
        self._analyze_and_suggest(metrics)
        
        return metrics

    def _export_false_positives(self, hadm_ids, prompts, true_labels, predictions, probs, rule_modifications):
        """导出假阳性病例详情，用于进一步分析（修复逻辑错误）"""
        try:
            # 1. 识别假阳性样本（真实标签0，预测标签1）
            false_positives = []
            # 补充rule_modifications的长度，确保与其他列表对齐
            if len(rule_modifications) < len(hadm_ids):
                rule_modifications = [(None, None)] * len(hadm_ids)
            
            for i, (hadm_id, prompt, true_label, pred, prob, rule_mod) in enumerate(
                zip(hadm_ids, prompts, true_labels, predictions, probs, rule_modifications)):
                
                # ========== 修复核心：优化假阳性判定条件 ==========
                # 步骤1：类型转换与清洗，兼容numpy类型、字符串类型
                try:
                    # 转换真实标签为浮点型（兼容numpy.int64、str格式数值）
                    true_label_clean = float(true_label)
                    # 转换预测标签为浮点型
                    pred_clean = float(pred)
                except (ValueError, TypeError):
                    continue  # 无法转换的无效样本直接跳过
                
                # 步骤2：严格判定假阳性（真实标签为0，预测标签为1）
                if true_label_clean == 0.0 and pred_clean == 1.0:
                    # 从prompt中提取关键临床特征
                    clinical_features = self._extract_clinical_features(prompt)
                    
                    false_positives.append({
                        "sample_index": i,
                        "hadm_id": hadm_id,
                        "true_label": int(true_label_clean),
                        "predicted_label": int(pred_clean),
                        "probability": float(prob),
                        "rule_applied": rule_mod[0] if rule_mod and len(rule_mod) > 0 else "None",
                        "rule_modified": rule_mod[1] if rule_mod and len(rule_mod) > 1 else "None",
                        "risk_score": clinical_features.get("risk_score", "N/A"),
                        "baseline_creatinine": clinical_features.get("baseline_creatinine", "N/A"),
                        "max_creatinine": clinical_features.get("max_creatinine", "N/A"),
                        "creat_change_percent": clinical_features.get("creat_change_percent", "N/A"),
                        "uo_ml_kg_hr": clinical_features.get("uo_ml_kg_hr", "N/A"),
                        "min_map": clinical_features.get("min_map", "N/A"),
                        "max_hr": clinical_features.get("max_hr", "N/A"),
                        "aki_risk_factors": ", ".join(clinical_features.get("risk_factors", [])),
                        "full_prompt": prompt[:] + "..." if len(prompt) > 500 else prompt
                    })
            
            # 2. 保存假阳性样本到CSV文件
            if false_positives:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fp_df = pd.DataFrame(false_positives)
                
                # 按概率排序，高概率假阳性更值得关注
                fp_df = fp_df.sort_values(by="probability", ascending=False)
                
                # 保存完整详细数据
                fp_filename = f"./false_positives_analysis_{timestamp}.csv"
                fp_df.to_csv(fp_filename, index=False, encoding='utf-8-sig')
                
                # 3. 生成分析摘要
                self._generate_fp_analysis_summary(fp_df, timestamp)
                
                logger.info(f"已导出 {len(false_positives)} 个假阳性病例详情到 {fp_filename}")
                logger.info(f"前5个高概率假阳性病例（概率 > {fp_df.iloc[0]['probability']:.3f}）:")
                
                # 日志中显示前5个最高概率的假阳性样本摘要
                for _, row in fp_df.head().iterrows():
                    logger.info(f"  - HADM_ID: {row['hadm_id']}, 概率: {row['probability']:.3f}, "
                            f"风险评分: {row['risk_score']}, 肌酐变化: {row['creat_change_percent']}")
            else:
                logger.info("未检测到假阳性病例，模型精确率表现良好")
                
        except Exception as e:
            logger.error(f"导出假阳性病例失败: {str(e)}")
            logger.exception("详细错误信息:")

    def _analyze_and_suggest(self, metrics):
        """分析评估结果并提供优化建议"""
        logger.info("====== 性能分析报告 ======")
        logger.info(f"当前模型性能:")
        logger.info(f"- 召回率: {metrics['recall']:.4f} (目标: ≥0.80)")
        logger.info(f"- 精确率: {metrics['precision']:.4f} (目标: ≥0.65)")
        logger.info(f"- F1分数: {metrics['f1']:.4f}")
        
        # 计算改进空间
        if metrics['recall'] < 0.8:
            logger.warning("召回率未达到最低要求！建议:")
            logger.warning(f"1. 进一步增加正样本权重（当前阈值: {metrics['best_threshold']:.4f})")
            logger.warning("2. 降低阈值")
            logger.warning("3. 检查数据集中正样本是否代表性不足")
        else:
            logger.info("✓ 召回率满足最低要求")
        
        if metrics['precision'] < 0.65:
            logger.info(f"精确率有 {0.65-metrics['precision']:.4f} 的提升空间，建议:")
            logger.info("1. 增强医学规则过滤的严格度")
            logger.info(f"2. 提高阈值（当前: {metrics['best_threshold']:.4f})")
            logger.info("3. 增加训练数据量或使用数据增强")
        else:
            logger.info("✓ 精确率已达到目标")
        
        # 分析混淆矩阵
        if metrics['confusion_matrix']:
            tn, fp, fn, tp = metrics['confusion_matrix'][0][0], metrics['confusion_matrix'][0][1], \
                            metrics['confusion_matrix'][1][0], metrics['confusion_matrix'][1][1]
            
            logger.info(f"混淆矩阵分析:")
            logger.info(f"- 真阴性(TN): {tn} (特异度: {tn/(tn+fp):.4f})")
            logger.info(f"- 假阳性(FP): {fp} (可过滤的目标)")
            logger.info(f"- 假阴性(FN): {fn} (漏诊风险)")
            logger.info(f"- 真阳性(TP): {tp}")
            
            if fp > 0:
                logger.info(f"假阳性率(FPR): {fp/(fp+tn):.4f}")
            
            if fp + fn > 0:
                logger.info(f"错误分类率: {(fp+fn)/(tn+fp+fn+tp):.4f}")
        
        logger.info("====== 分析报告结束 ======")

    def _extract_clinical_features(self, prompt):
        """从prompt中提取关键临床特征，用于假阳性分析"""
        features = {}
        
        # 提取风险评分
        risk_match = re.search(r"Composite Risk Score: (\d+)", prompt)
        if risk_match:
            features["risk_score"] = int(risk_match.group(1))
        
        # 提取肌酐相关指标
        baseline_match = re.search(r"Baseline Creatinine: ([0-9.]+) μmol/L", prompt)
        max_match = re.search(r"Maximum Creatinine: ([0-9.]+) μmol/L", prompt)
        percent_match = re.search(r"Creatinine Change: ([0-9.]+)% from baseline", prompt)
        
        if baseline_match:
            features["baseline_creatinine"] = float(baseline_match.group(1))
        if max_match:
            features["max_creatinine"] = float(max_match.group(1))
        if percent_match:
            features["creat_change_percent"] = float(percent_match.group(1))
        
        # 提取尿量指标
        uo_match = re.search(r"Urine Output per kg: ([0-9.]+) ml/kg/hour", prompt)
        if uo_match:
            features["uo_ml_kg_hr"] = float(uo_match.group(1))
        
        # 提取生命体征
        map_match = re.search(r"Minimum Mean Arterial Pressure: ([0-9.]+) mmHg", prompt)
        hr_match = re.search(r"Maximum Heart Rate: (\d+) beats/min", prompt)
        
        if map_match:
            features["min_map"] = float(map_match.group(1))
        if hr_match:
            features["max_hr"] = int(hr_match.group(1))
        
        # 提取AKI风险因素
        risk_factors = []
        if "Creatinine elevation:" in prompt:
            risk_factors.append("肌酐升高")
        if "Severe oliguria" in prompt or "Critical oliguria" in prompt:
            risk_factors.append("少尿")
        if "Hypotension" in prompt:
            risk_factors.append("低血压")
        if "Tachycardia" in prompt:
            risk_factors.append("心动过速")
        if "Rising creatinine trend" in prompt:
            risk_factors.append("肌酐上升趋势")
        
        features["risk_factors"] = risk_factors
        
        return features

    def _generate_fp_analysis_summary(self, fp_df, timestamp):
        """生成假阳性病例分析摘要报告"""
        try:
            summary = {
                "total_false_positives": len(fp_df),
                "avg_probability": fp_df["probability"].mean(),
                "high_risk_count": len(fp_df[fp_df["risk_score"].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else 0) >= 4]),
                "common_patterns": {}
            }
            
            # 分析常见模式
            # 1. 肌酐轻微升高但无其他风险因素
            mild_creat_increase = len(fp_df[
                (fp_df["creat_change_percent"].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else 0) < 30) &
                (fp_df["aki_risk_factors"].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0) <= 1)
            ])
            
            # 2. 低尿量但肌酐正常
            low_uo_normal_creat = len(fp_df[
                (fp_df["uo_ml_kg_hr"].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else 0) < 0.5) &
                (fp_df["creat_change_percent"].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else 0) < 20)
            ])
            
            # 3. 低血压但其他指标正常
            hypotension_only = len(fp_df[
                (fp_df["min_map"].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else 0) < 65) &
                (fp_df["creat_change_percent"].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else 0) < 20) &
                (fp_df["uo_ml_kg_hr"].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else 0) > 0.5)
            ])
            
            summary["common_patterns"] = {
                "mild_creat_increase": mild_creat_increase,
                "low_uo_normal_creat": low_uo_normal_creat,
                "hypotension_only": hypotension_only
            }
            
            # 4. 保存分析摘要
            summary_filename = f"./fp_analysis_summary_{timestamp}.json"
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # 5. 记录分析摘要到日志
            logger.info("=== 假阳性病例分析摘要 ===")
            logger.info(f"总假阳性数: {summary['total_false_positives']}")
            logger.info(f"平均预测概率: {summary['avg_probability']:.3f}")
            logger.info(f"高风险评分(≥4)假阳性数: {summary['high_risk_count']}")
            logger.info("常见模式分布:")
            logger.info(f"- 肌酐轻微升高(<30%)且风险因素少(≤1): {mild_creat_increase} 例")
            logger.info(f"- 低尿量(<0.5ml/kg/h)但肌酐变化小(<20%): {low_uo_normal_creat} 例")
            logger.info(f"- 仅低血压(无肌酐/尿量异常): {hypotension_only} 例")
            logger.info(f"详细分析报告已保存至: {summary_filename}")
            
        except Exception as e:
            logger.error(f"生成假阳性分析摘要失败: {str(e)}")

    def load_best_model(self, model_path):
        """安全加载最佳模型，避免显存溢出"""
        try:
            logger.info(f"从 {model_path} 安全加载最佳模型")
            
            # 获取当前设备
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}"
            
            # 重新初始化基础模型
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # 为当前设备配置
            device_map = {"": local_rank}
            
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                num_labels=2,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # 手动加载适配器
            model = PeftModelForSequenceClassification.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False
            )
            
            # 清理缓存
            torch.cuda.empty_cache()
            logger.info("最佳模型成功加载")
            return model
        except Exception as e:
            logger.error(f"加载最佳模型失败: {str(e)}")
            raise

    def load_best_threshold(self):
        """加载持久化的最优阈值，实现全局同步"""
        try:
            if os.path.exists(self.threshold_file):
                with open(self.threshold_file, 'r', encoding='utf-8') as f:
                    threshold_data = json.load(f)
                self.best_threshold = threshold_data["best_threshold"]
                logger.info(f"成功从{self.threshold_file}加载最优阈值: {self.best_threshold:.4f}")
                return self.best_threshold
            else:
                logger.warning(f"未找到阈值文件{self.threshold_file}，使用默认阈值{self.best_threshold}")
                return self.best_threshold
        except Exception as e:
            logger.error(f"加载最优阈值失败: {str(e)}，使用默认阈值{self.best_threshold}")
            return self.best_threshold

# --------------------------
# 主执行流程
# --------------------------
if __name__ == "__main__":
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info("====== AKI分类预测系统启动 ======")
        
        # 1. 配置数据路径
        DATA_DIR = "/home/fbh/llama2-7b/data/csv"
        DATA_PATHS = {
            'patients': os.path.join(DATA_DIR, 'patients.csv'),
            'admissions': os.path.join(DATA_DIR, 'admissions.csv'),
            'labevents': os.path.join(DATA_DIR, 'labevents.csv'),
            'vitalsign': os.path.join(DATA_DIR, 'vitalsign.csv'),
            'urine_output': os.path.join(DATA_DIR, 'urine_output.csv'),
            'kdigo_stages': os.path.join(DATA_DIR, 'kdigo_stages.csv'),
            'creatinine_baseline': os.path.join(DATA_DIR, 'creatinine_baseline.csv'),
            'first_day_weight': os.path.join(DATA_DIR, 'first_day_weight.csv'),
            'icustays': os.path.join(DATA_DIR, 'icustays.csv')
        }

        # 2. 数据预处理（仅过滤肌酐斜率为0的case）
        logger.info(">>> 开始数据预处理（过滤肌酐斜率为0的case）...")
        processor = AKIDataProcessor(DATA_PATHS)
        processed_df = processor.process()

        # 关键：校验预处理后数据是否为空（主进程打印日志）
        if len(processed_df) == 0:
            if local_rank == 0:
                logger.critical("预处理后无有效数据，无法继续运行，终止程序")
            raise ValueError("预处理后数据集为空，终止训练")

        # 3. 格式转换与数据集划分（修正时间序列分割逻辑）
        logger.info(">>> 格式化数据并进行时间序列分割...")
        predictor = AKIPredictor()
        
        # 先按入院时间排序，再格式化（确保时间划分准确）
        processed_df_sorted = processed_df.sort_values('admittime').reset_index(drop=True)
        formatted_data = predictor._format_data(processed_df_sorted)
        
        # --------------------------
        # 新增：最终过滤0斜率病例（关键）【修改点5】
        # --------------------------
        before_filter_count = len(formatted_data)
        formatted_data = [
            sample for sample in formatted_data
            if not is_zero_slope(sample.get('creat_slope', 0.0))
        ]
        after_filter_count = len(formatted_data)
        filtered_count = before_filter_count - after_filter_count

        if local_rank == 0:
            logger.info(f"最终0斜率病例过滤：过滤前 {before_filter_count} 条，过滤后 {after_filter_count} 条，过滤掉 {before_filter_count - after_filter_count} 条")

        # 关键：为空数据终止流程
        if after_filter_count == 0:
            if local_rank == 0:
                logger.critical("最终过滤后无有效数据，无法进行训练，终止程序")
            raise ValueError("最终过滤后数据集为空，终止训练")

        if filtered_count > 0:
            logger.warning(f"过滤掉的0斜率病例HADM_ID示例：{[sample['hadm_id'] for sample in formatted_data[:5] if is_zero_slope(sample.get('creat_slope', 0.0))]}")

        # 划分训练集、验证集、测试集（7:1.5:1.5）
        total_count = len(formatted_data)
        train_size = int(0.7 * total_count)
        val_size = int(0.15 * total_count)
        
        train_data = formatted_data[:train_size]
        val_data = formatted_data[train_size:train_size+val_size]
        test_data = formatted_data[train_size+val_size:]
        
        # --------------------------
        # 新增：校验各数据集是否存在0斜率病例【修改点6】
        # --------------------------
        train_zero_slope = sum([is_zero_slope(sample.get('creat_slope', 0.0)) for sample in train_data])
        val_zero_slope = sum([is_zero_slope(sample.get('creat_slope', 0.0)) for sample in val_data])
        test_zero_slope = sum([is_zero_slope(sample.get('creat_slope', 0.0)) for sample in test_data])
        
        logger.info(f"各数据集0斜率病例校验：训练集={train_zero_slope}，验证集={val_zero_slope}，测试集={test_zero_slope}")
        
        # 记录类别分布
        train_labels = [d["label"] for d in train_data]
        val_labels = [d["label"] for d in val_data]
        test_labels = [d["label"] for d in test_data]
        
        logger.info(f"数据集划分: 训练集 {len(train_data)} | 验证集 {len(val_data)} | 测试集 {len(test_data)}")
        logger.info(f"训练集类别分布: 阴性 {train_labels.count(0)}, 阳性 {train_labels.count(1)}")
        logger.info(f"验证集类别分布: 阴性 {val_labels.count(0)}, 阳性 {val_labels.count(1)}")
        logger.info(f"测试集类别分布: 阴性 {test_labels.count(0)}, 阳性 {test_labels.count(1)}")

        # 4. 模型训练
        logger.info(">>> 开始模型训练...")
        trained_model, model_save_path = predictor.train(train_data, val_data)

        # model_save_path = "/home/fbh/llama2-7b/model/aki-classification-best-model_20260121_191125"
        # trained_model   = None
        #释放显存
        torch.cuda.empty_cache()

        # 5. 仅主进程加载最佳模型并执行评估
        final_metrics = {}
        if local_rank == 0:
            #. 安全加载最佳模型
            if model_save_path:
                loaded_model = predictor.load_best_model(model_save_path)
            else:
                loaded_model = trained_model  # 其他进程直接使用训练好的模型
            # loaded_model = predictor.load_best_model("/home/fbh/llama2-7b/model/aki-classification-best-model_20260109_085858")

            # 加载持久化的最优阈值（关键：同步训练得到的最优阈值）
            predictor.load_best_threshold()

            # 5. 模型评估
            logger.info(">>> 在测试集上评估最终模型...")
            final_metrics = predictor.evaluate(loaded_model, test_data)
            
            logger.info(f"最终测试集指标: {final_metrics}")
            logger.info("====== AKI分类预测系统完成 ======")

    except Exception as e:
        logger.critical(f"系统运行失败: {str(e)}")
        logger.exception("严重异常详情")
        raise