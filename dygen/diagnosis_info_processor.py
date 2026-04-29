import random
random.seed(42)

from aiotinydb import AIOTinyDB
from tinydb import Query, TinyDB
from pydantic import BaseModel
import traceback

from .gen_prompts import SIMILAR_DIAGNOSIS_PROMPT, SIMILAR_DIAGNOSIS_SYSTEM_PROMPT

from utils import ModelCaller, run_with_semaphore

# 使用doubao 联网 agent
class DiagnosisInfoProcessor:
    def __init__(self, 
                 worker_model, 
                 web_model,
                 random_temp, 
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
        self.web_model = web_model
        self.random_temp = random_temp
        self.knowledge_path = knowledge_path

        self.QueryModel = Query()
        self._init_caller()

    def _init_caller(self):
        caller = ModelCaller()
        self.call_api_json_web = caller.call_api_json_web

    class DiagnosisInfo(BaseModel):
        class RootDiagnosis(BaseModel):
            name: str
            symptoms: list[str]
        
        class SimilarDiagnosis(BaseModel):
            name: str
            symptoms: list[str]
        
        root_diagnosis: RootDiagnosis
        similar_diagnoses: list[SimilarDiagnosis]

    async def process_diagnosis(
            self, 
            diagnosis_name: str,
            overwrite: bool = False
        ):
        try:

            # 使用LLM总结
            # 检查是否已经有候选诊断信息，如果已有，跳过 _get_candidates
            cur_similar_diagnoses = self.similar_diagnoses_table.get(self.QueryModel.diagnosis_name == diagnosis_name)
            # 如果两项信息都存在，跳过，缺少任意一项信息则进行候选诊断的获取
            if not cur_similar_diagnoses or overwrite:
                # 如果没有候选诊断信息，则进行候选诊断的获取
                system_prompt = SIMILAR_DIAGNOSIS_SYSTEM_PROMPT
                prompt = SIMILAR_DIAGNOSIS_PROMPT.format(
                    root_diagnosis=diagnosis_name,
                    n=10
                )
                diagnosis_info, references = await self.call_api_json_web(
                    model=self.web_model,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    temperature=self.random_temp,
                    json_schema=self.DiagnosisInfo,
                )

                self.similar_diagnoses_table.upsert({
                    'diagnosis_name': diagnosis_name,
                    'refer_diagnosis_symptoms': diagnosis_info['root_diagnosis']['symptoms'],
                    'similar_diagnoses': diagnosis_info['similar_diagnoses']
                }, self.QueryModel.diagnosis_name == diagnosis_name)
                self.references_table.upsert({
                    'diagnosis_name': diagnosis_name,
                    'references': references
                }, self.QueryModel.diagnosis_name == diagnosis_name)

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
            self.references_table = db.table('similar_diagnoses_references')
            tasks = []

            # 收集所有处理任务
            for id, diag in enumerate(diagnosis_lst):
                # print(diag)
                if self.similar_diagnoses_table.contains(self.QueryModel.diagnosis_name == diag) and not overwrite:
                    continue
                tasks.append(self.process_diagnosis(
                    diagnosis_name=diag,
                    overwrite=overwrite
                    ))

            # 并发执行所有任务
            await run_with_semaphore(tasks, max_concurrency,
                                               desc='[DiagnosisInfoProcessor] Searching similar diagnoses')
            # await tqdm_asyncio.gather(*tasks, total=len(tasks), 
            #                           desc='[DiagnosisInfoProcessor] Searching candidate diagnoses')