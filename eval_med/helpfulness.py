import asyncio
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel, Field

from typing import List, Dict, Tuple, Literal
from itertools import chain

from utils import ModelCaller, run_with_semaphore

from .eval_prompts import (
    SCORE_SYSTEM_PROMPT, SCORE_PROMPT,
)

import numpy as np

import os

from aiotinydb import AIOTinyDB
from tinydb import Query
import asyncio
from typing import List, Dict

from copy import deepcopy

class HelpfulnessScore(BaseModel):
    class RealDiagnosisScore(BaseModel):
        diagnosis_in_response: List[str]
        reason: str
        score: Literal[100, 50, 0] = Field(..., description='是否给出正确诊断（100分）')

    class DiagnosisEvidencesScore(BaseModel):
        reason: str
        score: Literal[100, 50, 0] = Field(..., description='是否给出核心诊断依据（100分）')

    class TreatmentSuggestionsScore(BaseModel):
        reason: str
        score: Literal[100, 50, 0] = Field(..., description='是否给出核心治疗建议（100分）')

    class LifestyleSuggestionsScore(BaseModel):
        reason: str
        score: Literal[100, 50, 0] = Field(..., description='是否给出核心生活方式建议（100分）')

    real_diagnosis_score: RealDiagnosisScore
    diagnosis_evidences_score: DiagnosisEvidencesScore
    treatment_suggestions_score: TreatmentSuggestionsScore
    lifestyle_suggestions_score: LifestyleSuggestionsScore

class HelpfulnessEvaluator:
    def __init__(
            self, 
            logs_dir: str, 
            predictions_path: str,
            agent_model: str, 
            decision_temp: float = 0., 
            **kwargs):
        """
        初始化 HelpfulnessEvaluator 类

        :param logs_dir: 日志目录
        :param predictions_path: 预测结果文件路径
        :param agent_model: 用于评分的模型
        :param decision_temp: 随机温度值，用于控制生成的多样性
        """
        self.logs_dir = logs_dir
        self.predictions_path = predictions_path
        self.logs_path = os.path.join(self.logs_dir, os.path.split(self.predictions_path)[-1])

        self.agent_model = agent_model
        self.decision_temp = decision_temp

        self.QueryModel = Query()
        self._init_caller()

    def _init_caller(self):
        """
        初始化 API 调用器
        """
        caller = ModelCaller()
        self.call_api_json = caller.call_api_json

    async def process_sample(self, sample: Dict, overwrite: bool = False):
        """
        处理单个样本，并将评分结果保存到数据库。

        Args:
            sample (Dict): 样本数据
            overwrite (bool): 是否覆盖现有评分结果

        Returns:
            Dict: 当前样本的评分结果
        """
        question_id = sample['question_id']

        if not sample:
            print(f"sample {question_id} not found in database.")
            return None

        # 如果已有结果且不允许覆盖，则跳过
        score_result = self.logs_table.get(self.QueryModel.question_id == question_id)
        if overwrite or not score_result:
            response = sample['prediction']
            question = sample['question']
            score_point = sample['score_points'] 

            sys_prompt = SCORE_SYSTEM_PROMPT
            prompt = SCORE_PROMPT.format(
                question=question,
                response=response,
                real_diagnosis=score_point['refer_diagnosis'],
                diagnosis_evidences=score_point['diagnosis_evidences'][:3],
                treatment_suggestions=score_point['treatment_suggestions'][:3],
                lifestyle_suggestions=score_point['lifestyle_suggestions'][:3],
            )

            # 调用 API 进行评分
            try:
                score_result = await self.call_api_json(
                    model=self.agent_model,
                    system_prompt=sys_prompt,
                    max_tokens=8192,
                    prompt=prompt,
                    temperature=self.decision_temp,
                    json_schema=HelpfulnessScore
                )

                # 保存结果到数据库
                save_result = deepcopy(score_result)
                save_result['question_id'] = question_id
                save_result['patient_id'] = sample['patient_id']
                save_result['prediction_id'] = sample['prediction_id']
                self.logs_table.upsert(save_result, self.QueryModel.question_id == question_id)
                return score_result

            except Exception as e:
                print(f"Error processing sample {question_id}: {e}")
                return None
        else:
            # 加一个await sleep，用来更新进度条
            await asyncio.sleep(0)
            return score_result

    async def run(self, overwrite: bool = False, max_concurrency: int = 8, position=0) -> Dict:
        """
        遍历数据库中的样本，处理评分并保存结果。

        Args:
            overwrite (bool): 是否覆盖已有评分结果
            max_concurrency (int): 最大并发数
            position (int): 进度条位置

        Returns:
            Dict: 包含所有评分结果和统计信息
        """
        async with AIOTinyDB(self.predictions_path, indent=4, separators=(',', ': '), ensure_ascii=False) as predictions_db, \
                AIOTinyDB(self.logs_path, indent=4, separators=(',', ': '), ensure_ascii=False) as logs_db:
            self.predictions_table = predictions_db.table('predictions')
            self.logs_table = logs_db.table('helpfulness')

            if overwrite:
                self.logs_table.truncate()

            tasks = []

            for i, sample in enumerate(self.predictions_table):
                if sample['question_id'] == 'DxBench_1001_misplaced':
                    continue
                tasks.append(self.process_sample(sample, overwrite=overwrite))
            
            results = await run_with_semaphore(tasks, max_concurrency, desc='[HelpfulnessEvaluator] Processing samples', position=position)

            # 计算统计分数
            scores = self.compute_scores(results)

        return scores

    def compute_scores(self, score_results: List[Dict]) -> Dict:
        """
        计算分数并生成统计结果。

        Args:
            score_results (List[Dict]): 包含每个类别分数的记录列表。

        Returns:
            Dict: 包括real_diagnosis_score和helpfulness_score的统计信息。
        """
        # 所有类别的分数
        category_scores = {
            'real_diagnosis_score': [],
            'diagnosis_evidences_score': [],
            'treatment_suggestions_score': [],
            'lifestyle_suggestions_score': []
        }

        # helpfulness相关的三个类别（不包括real_diagnosis_score）
        helpfulness_categories = ['diagnosis_evidences_score', 'treatment_suggestions_score', 'lifestyle_suggestions_score']
        helpfulness_total_scores = []

        for score_record in score_results:
            # 收集所有类别的分数
            for category in category_scores.keys():
                score = score_record.get(category, {}).get('score', 0)
                category_scores[category].append(score)
            
            # 计算helpfulness总分（不包括real_diagnosis_score）
            helpfulness_score = sum([
                score_record.get(category, {}).get('score', 0) 
                for category in helpfulness_categories
            ])
            helpfulness_total_scores.append(helpfulness_score)

        # 计算每个类别的平均分
        bound = {
            'real_diagnosis_score': 100,  # real_diagnosis_score满分100
            'diagnosis_evidences_score': 100,
            'treatment_suggestions_score': 100,
            'lifestyle_suggestions_score': 100
        }
        
        ## 映射到0-100分
        category_averages = {
            category: (sum(scores) / len(scores)) / bound[category] * 100 if scores else 0
            for category, scores in category_scores.items()
        }

        # 单独计算real_diagnosis_score平均分
        real_diagnosis_percentage = category_averages['real_diagnosis_score']

        # 计算helpfulness总得分百分比（三个类别总分为300分）
        helpfulness_percentage = (sum(helpfulness_total_scores) / len(helpfulness_total_scores)) / 300 * 100 if helpfulness_total_scores else 0

        # helpfulness三个类别的小分
        helpfulness_specific_scores = {
            category: category_averages[category]
            for category in helpfulness_categories
        }

        return {
            'real_diagnosis_score': real_diagnosis_percentage,
            'final_score': helpfulness_percentage,
            'specific_scores': helpfulness_specific_scores,
        }
