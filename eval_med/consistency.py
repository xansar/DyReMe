import asyncio
from tqdm.asyncio import tqdm_asyncio

from pydantic import BaseModel
import os
from typing import List, Dict, Tuple

from utils import ModelCaller, run_with_semaphore
from .eval_prompts import (
    CONSISTENSY_RESULT_EXTRACT_SYSTEM_PROMPT, CONSISTENSY_RESULT_EXTRACT_PROMPT,
    STANDARDIZE_SYSTEM_PROMPT, STANDARDIZE_PROMPT
)

from itertools import chain
from collections import Counter
from scipy.stats import entropy

import numpy as np

from aiotinydb import AIOTinyDB
from tinydb import Query



class DiagnosesList(BaseModel):
    diagnoses: List[str]


class ConsistencyEvaluator:
    def __init__(
            self, 
            logs_dir: str, 
            predictions_path: str,
            agent_model: str, 
            decision_temp: float = 0., 
            **kwargs):
        """
        初始化 ConsistencyEvaluator 类

        :param db_path: 数据库文件路径
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

    class StandardizedDiagnoses(BaseModel):
        diagnosis_1: str
        diagnosis_2: str
        diagnosis_3: str
        diagnosis_4: str
        # diagnosis_5: str
        # diagnosis_6: str

    async def process_patient(self, patient_id: str, overwrite: bool = False):
        """
        处理单个样本，并将评分结果保存到数据库。

        Args:
            sample_id (int): 样本 ID
            overwrite (bool): 是否覆盖现有评分结果

        Returns:
            Dict: 当前样本的评分结果
        """

        # 如果已有结果且不允许覆盖，则跳过
        cur_result = self.consistency_logs_table.get(self.QueryModel.patient_id == patient_id)
        if overwrite or not cur_result:
            # 寻找一个病人对应的一组诊断结果
            help_log_results = self.helpfulness_logs_table.search(self.QueryModel.patient_id == patient_id)
            # 提取诊断结果
            diag_results = [log['real_diagnosis_score']["diagnosis_in_response"] for log in help_log_results]
            assert len(diag_results) == 4, f"Diagnosis results for patient {patient_id} is not complete."
            # 标准化诊断结果
            standardized_diagnoses_dict = await self.call_api_json(
                model=self.agent_model,
                system_prompt=STANDARDIZE_SYSTEM_PROMPT,
                prompt=STANDARDIZE_PROMPT.format(
                    diagnoses=[f'raw_diagnosis_{i + 1} = "{d}"\n' for i, d in enumerate(diag_results)]
                ),
                temperature=self.decision_temp,
                json_schema=self.StandardizedDiagnoses
            )

            # 保存标准化结果
            standardized_diagnoses_lst = [
                standardized_diagnoses_dict[f'diagnosis_{i + 1}'] for i in range(4)
            ]
            save_result = {
                'patient_id': patient_id,
                'raw_diagnoses': diag_results,
                'standardized_diagnoses': standardized_diagnoses_lst
            }
            self.consistency_logs_table.upsert(save_result, self.QueryModel.patient_id == patient_id)
        else:
            await asyncio.sleep(0)
            save_result = cur_result
        
        return save_result



    async def run(self, overwrite: bool = False, max_concurrency: int = 8, position=0):
        """
        遍历数据库中的样本，处理评分并保存结果。

        Args:
            overwrite (bool): 是否覆盖已有评分结果

        Returns:
            Dict: 包含所有评分结果和统计信息
        """
        # print('running')
        async with AIOTinyDB(self.predictions_path, indent=4, separators=(',', ': '), ensure_ascii=False) as predictions_db, \
                AIOTinyDB(self.logs_path, indent=4, separators=(',', ': '), ensure_ascii=False) as logs_db:
            self.predictions_table = predictions_db.table('predictions')
            self.consistency_logs_table = logs_db.table('consistency')
            self.helpfulness_logs_table = logs_db.table('helpfulness')

            tasks = []

            patient_ids = set([sample['patient_id'] for sample in self.predictions_table])

            for i, patient_id in enumerate(patient_ids):

                tasks.append(self.process_patient(patient_id, overwrite=overwrite))

            # results = await tqdm_asyncio.gather(
            #     *tasks, desc='[ConsistencyEvaluator] Processing samples', ncols=75)
            
            results = await run_with_semaphore(tasks, max_concurrency, desc='[ConsistencyEvaluator] Processing samples', position=position)

            # 计算统计分数

            scores = self._compute_scores(results)

        return scores
    
    def _compute_scores(self, sample: List[Dict]) -> List[float]:
        # 统计每个instance中不同诊断结果的数量
        # 计算每个instance的诊断结果的熵
        def _compute_entropy_for_instance(diagnoses):
            counter = Counter(diagnoses)
            total = len(diagnoses)
            probs = [v / total for v in counter.values()]
            return entropy(probs, base=2).item()
        
        diagnoses_lst = [s['standardized_diagnoses'] for s in sample]

        entropies = [_compute_entropy_for_instance(d) for d in diagnoses_lst]
        mean_entropies = np.mean(entropies).item()

        final_score = ((1 - mean_entropies / np.log2(4)) * 100).item()

        return {
            'final_score': final_score,
            'specific_scores': mean_entropies,
        }