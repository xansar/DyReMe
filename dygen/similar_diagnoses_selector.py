import random
random.seed(42)

from aiotinydb import AIOTinyDB
from tinydb import Query, TinyDB
from pydantic import BaseModel
import traceback

from .gen_prompts import JUDGE_SYMPTOMS_OVERCAPTION_SYSTEM_PROMPT, JUDGE_SYMPTOMS_OVERCAPTION_PROMPT

from utils import ModelCaller, run_with_semaphore


class SimilarDiagnosisSelector:
    def __init__(self, 
                 worker_model, 
                 decision_temp, 
                 knowledge_path, 
                 ):
        """
        初始化诊断信息处理器
        :param diagnosis_lst: 需要检索的疾病列表
        :param worker_model: 用于生成问题的模型
        :param random_temp: 随机温度值，用于控制生成的多样性
        :param knowledge_path: 疾病数据库路径
        """
        self.worker_model = worker_model
        self.decision_temp = decision_temp
        self.knowledge_path = knowledge_path

        self.QueryModel = Query()
        self._init_caller()

    def _init_caller(self):
        caller = ModelCaller()
        # self.call_api_json_web = caller.call_api_json_web
        self.call_api_json = caller.call_api_json

    class DifferentialSymptoms(BaseModel):
        differential_symptoms: list[str]


    async def process_diagnosis(
            self, 
            diagnosis_name: str,
            symptoms_lst: list[str],
            overwrite: bool = False
        ):
        try:

            # 使用LLM总结
            # 检查是否已经有干扰诊断信息，如果没有，报错
            cur_similar_diagnoses = self.similar_diagnoses_table.get(self.QueryModel.diagnosis_name == diagnosis_name)
            if cur_similar_diagnoses:
                # 如果有干扰诊断信息，从中抽取能够满足条件的干扰诊断
                # 条件：至少存在一个symptoms_lst中的症状不属于干扰诊断的症状列表
                # shuffle the similar diagnoses to ensure randomness
                random.shuffle(cur_similar_diagnoses['similar_diagnoses'])
                # 遍历所有相似诊断
                # 如果没有症状列表，则报错
                for similar_diag in cur_similar_diagnoses['similar_diagnoses']:
                    similar_diag_symptoms = similar_diag['symptoms']
                    system_prompt = JUDGE_SYMPTOMS_OVERCAPTION_SYSTEM_PROMPT
                    prompt = JUDGE_SYMPTOMS_OVERCAPTION_PROMPT.format(
                        symptoms_lst_A=symptoms_lst,
                        symptoms_lst_B=similar_diag_symptoms
                    )
                    differential_symptoms = await self.call_api_json(
                        model=self.worker_model,
                        system_prompt=system_prompt,
                        prompt=prompt,
                        temperature=self.decision_temp,
                        json_schema=self.DifferentialSymptoms,
                    )
                    # 确保differential_symptoms中元素不在symptoms_lst中，但是在similar_diag_symptoms中
                    valid_differential_symptoms = [
                        symptom for symptom in differential_symptoms['differential_symptoms']
                        if symptom not in symptoms_lst and symptom in similar_diag_symptoms
                    ]
                    if valid_differential_symptoms:
                        # 如果存在满足条件的干扰诊断，直接返回
                        print(f"Found valid differential symptoms for {diagnosis_name}: {valid_differential_symptoms}")
                        return valid_differential_symptoms
                # 如果没有找到满足条件的干扰诊断，报错
                raise ValueError(f"No valid differential diagnoses found for {diagnosis_name} with symptoms {symptoms_lst}")
            else:
                raise ValueError(f"No similar diagnoses found for {diagnosis_name}. Please run the diagnosis info processor first.")

        except Exception as e:
            print(f"Error processing diagnosis {diagnosis_name}: {e}")
            traceback.print_exc()

    async def run(self, diagnosis_lst, overwrite, max_concurrency=8): # 去重过的diagnosis_lst
        """
        根据实际诊断获取搜索结果
        :param real_diagnosis_lst: 实际诊断列表
        :param query_templates: 查询模板
        :return: 搜索结果列表
        """
        async with AIOTinyDB(self.knowledge_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db:
            self.similar_diagnoses_table = db.table('similar_diagnoses')
            tasks = []

            # 收集所有处理任务
            for id, diag in enumerate(diagnosis_lst):
                tasks.append(self.process_diagnosis(
                    diagnosis_name=diag,
                    overwrite=overwrite
                    ))

            # 并发执行所有任务
            await run_with_semaphore(tasks, max_concurrency,
                                               desc='[DiagnosisInfoProcessor] Searching similar diagnoses')
            # await tqdm_asyncio.gather(*tasks, total=len(tasks), 
            #                           desc='[DiagnosisInfoProcessor] Searching candidate diagnoses')