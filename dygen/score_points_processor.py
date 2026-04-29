from typing import List, Dict
from pydantic import Field
from itertools import chain

from aiotinydb import AIOTinyDB
from tinydb import Query
from pydantic import BaseModel
import traceback


from tqdm.asyncio import tqdm_asyncio

from .gen_prompts import (
    SCORE_POINTS_SEARCH_SYSTEM_PROMPT,
    DIAGNOSIS_EVIDENCES_GENERATE_PROMPT,
    TREATMENT_SUGGESTIONS_GENERATE_PROMPT,
    LIFESTYLE_SUGGESTIONS_GENERATE_PROMPT,
    SCORE_POINTS_GENERATE_PROMPT, 
    SCORE_POINTS_GENERATE_SYSTEM_PROMPT
)

from utils import ModelCaller, run_with_semaphore

class ScorePointsProcessor:
    def __init__(
            self, 
            generator_model, 
            web_model,
            decision_temp, 
            data_path, 
            knowledge_path, 
        ):
        """
        初始化诊断信息处理器
        :param worker_model: 用于生成问题的模型
        :param random_temp: 随机温度值，用于控制生成的多样性
        :param knowledge_path: 疾病数据库路径
        """
        self.generator_model = generator_model
        self.web_model = web_model
        self.decision_temp = decision_temp
        self.knowledge_path = knowledge_path
        self.data_path = data_path
        self._init_caller()

        self.QueryModel = Query()


    def _init_caller(self):
        caller = ModelCaller()
        self.call_api_json = caller.call_api_json
        self.call_api_json_web = caller.call_api_json_web

    class ScorePoints(BaseModel):
        diagnosis_evidences: List[str] = Field(..., description='确认真实诊断的核心依据')
        treatment_suggestions: List[str] = Field(..., description='治疗真实诊断的核心建议')
        lifestyle_suggestions: List[str] = Field(..., description='改善病情的核心生活方式建议')

    class DiagnosisEvidences(BaseModel):
        diagnosis_evidences: List[str] = Field(..., description='诊断依据')

    class TreatmentSuggestions(BaseModel):
        treatment_suggestions: List[str] = Field(..., description='治疗建议')

    class LifestyleSuggestions(BaseModel):
        lifestyle_suggestions: List[str] = Field(..., description='生活方式建议')
    
    async def _generate_diagnosis_evidences(self, diagnosis_name):
        """
        生成诊断依据
        """
        system_prompt = SCORE_POINTS_SEARCH_SYSTEM_PROMPT
        prompt = DIAGNOSIS_EVIDENCES_GENERATE_PROMPT.format(
            refer_diagnosis=diagnosis_name
        )
        
        results, references = await self.call_api_json_web(
            model=self.web_model,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=self.decision_temp,
            json_schema=self.DiagnosisEvidences
        )
        
        # 保存引用信息
        self.references_table_dict['diagnosis_evidences'].upsert({
            'diagnosis_name': diagnosis_name,
            'references': references
        }, self.QueryModel.diagnosis_name == diagnosis_name)

        if results and 'diagnosis_evidences' in results:
            return results['diagnosis_evidences']
        else:
            raise ValueError(f"Invalid diagnosis evidences results: {results}")

    async def _generate_treatment_suggestions(self, diagnosis_name):
        """
        生成治疗建议
        """
        system_prompt = SCORE_POINTS_SEARCH_SYSTEM_PROMPT
        prompt = TREATMENT_SUGGESTIONS_GENERATE_PROMPT.format(
            refer_diagnosis=diagnosis_name
        )
        
        results, references = await self.call_api_json_web(
            model=self.web_model,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=self.decision_temp,
            json_schema=self.TreatmentSuggestions
        )
        
        # 保存引用信息
        self.references_table_dict['treatment_suggestions'].upsert({
            'diagnosis_name': diagnosis_name,
            'references': references
        }, self.QueryModel.diagnosis_name == diagnosis_name)

        if results and 'treatment_suggestions' in results:
            return results['treatment_suggestions']
        else:
            raise ValueError(f"Invalid treatment suggestions results: {results}")

    async def _generate_lifestyle_suggestions(self, diagnosis_name):
        """
        生成生活方式建议
        """
        system_prompt = SCORE_POINTS_SEARCH_SYSTEM_PROMPT
        prompt = LIFESTYLE_SUGGESTIONS_GENERATE_PROMPT.format(
            refer_diagnosis=diagnosis_name
        )
        
        results, references = await self.call_api_json_web(
            model=self.web_model,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=self.decision_temp,
            json_schema=self.LifestyleSuggestions
        )
        
        # 保存引用信息
        self.references_table_dict['lifestyle_suggestions'].upsert({
            'diagnosis_name': diagnosis_name,
            'references': references
        }, self.QueryModel.diagnosis_name == diagnosis_name)

        if results and 'lifestyle_suggestions' in results:
            return results['lifestyle_suggestions']
        else:
            raise ValueError(f"Invalid lifestyle suggestions results: {results}")


    async def process_question(
            self, 
            question_doc_id: int,
            overwrite: bool = False
        ):
        try:
            question = self.questions_table.get(doc_id=question_doc_id)
            question_id = question['question_id']
            diagnosis_name = question['refer_diagnosis']

            # 需要确保问题已经经过验证和润色，对于没有经过验证和润色的问题，跳过
            cur_logs = self.logs_table.get(self.QueryModel.question_id == question_id)
            # cur_logs如果为空，或者最后一个log的result不为true，跳过
            if not cur_logs or not cur_logs['logs'][-1]['result']:
                return

            # 分别生成诊断依据和治疗建议
            # 检查是否已经有得分点
            cur_score_points = self.score_points_table.get(self.QueryModel.question_id == question_id)
            if not cur_score_points or overwrite:
                # 检查是否已有该诊断的诊断依据，如果没有则生成
                diagnosis_evidences_cache = self.diagnosis_evidences_table.get(
                    self.QueryModel.diagnosis_name == diagnosis_name
                )
                if not diagnosis_evidences_cache or overwrite:
                    diagnosis_evidences = await self._generate_diagnosis_evidences(diagnosis_name)
                    self.diagnosis_evidences_table.upsert({
                        'diagnosis_name': diagnosis_name,
                        'evidences': diagnosis_evidences
                    }, self.QueryModel.diagnosis_name == diagnosis_name)
                else:
                    diagnosis_evidences = diagnosis_evidences_cache['evidences']
                
                # 检查是否已有该诊断的治疗建议，如果没有则生成
                treatment_suggestions_cache = self.treatment_suggestions_table.get(
                    self.QueryModel.diagnosis_name == diagnosis_name
                )
                if not treatment_suggestions_cache or overwrite:
                    treatment_suggestions = await self._generate_treatment_suggestions(diagnosis_name)
                    self.treatment_suggestions_table.upsert({
                        'diagnosis_name': diagnosis_name,
                        'suggestions': treatment_suggestions
                    }, self.QueryModel.diagnosis_name == diagnosis_name)
                else:
                    treatment_suggestions = treatment_suggestions_cache['suggestions']
                
                # 检查是否已有该诊断的生活方式建议，如果没有则生成
                lifestyle_suggestions_cache = self.lifestyle_suggestions_table.get(
                    self.QueryModel.diagnosis_name == diagnosis_name
                )
                if not lifestyle_suggestions_cache or overwrite:
                    lifestyle_suggestions = await self._generate_lifestyle_suggestions(diagnosis_name)
                    self.lifestyle_suggestions_table.upsert({
                        'diagnosis_name': diagnosis_name,
                        'suggestions': lifestyle_suggestions
                    }, self.QueryModel.diagnosis_name == diagnosis_name)
                else:
                    lifestyle_suggestions = lifestyle_suggestions_cache['suggestions']
                
                # 组合得分点
                score_points = {
                    'diagnosis_evidences': diagnosis_evidences,
                    'treatment_suggestions': treatment_suggestions,
                    'lifestyle_suggestions': lifestyle_suggestions,
                    'refer_diagnosis': question['refer_diagnosis']
                }

                self.score_points_table.upsert({
                    'question_id': question_id,
                    'score_points': score_points
                },
                self.QueryModel.question_id == question_id
                )

        except Exception as e:
            print(f"Error processing patient {question_doc_id}: {e}")
            traceback.print_exc()

    async def run(self, overwrite, max_concurrency = 8): # 去重过的diagnosis_lst
        """
        根据实际诊断获取搜索结果
        :param real_diagnosis_lst: 实际诊断列表
        :param query_templates: 查询模板
        :return: 搜索结果列表
        """
        async with AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db, \
            AIOTinyDB(self.knowledge_path, indent=4, separators=(',', ': '), ensure_ascii=False) as knowledge_db:
            self.questions_table = db.table('questions')
            self.score_points_table = db.table('score_points')
            self.logs_table = db.table('logs')

            # 缓存表，存储已生成的诊断依据、治疗建议和生活方式建议
            self.diagnosis_evidences_table = knowledge_db.table('diagnosis_evidences')
            self.treatment_suggestions_table = knowledge_db.table('treatment_suggestions')
            self.lifestyle_suggestions_table = knowledge_db.table('lifestyle_suggestions')
            
            # 分别存储诊断依据、治疗建议和生活方式建议的引用信息
            self.references_table_dict = {
                'diagnosis_evidences': knowledge_db.table('diagnosis_evidences_references'),
                'treatment_suggestions': knowledge_db.table('treatment_suggestions_references'),
                'lifestyle_suggestions': knowledge_db.table('lifestyle_suggestions_references')
            }
            
            if overwrite:
                self.score_points_table.truncate()
                self.diagnosis_evidences_table.truncate()
                self.treatment_suggestions_table.truncate()
                self.lifestyle_suggestions_table.truncate()
                for key in self.references_table_dict.keys():
                    self.references_table_dict[key].truncate()
                

            tasks = []

            # 收集所有处理任务
            for i in range(len(self.questions_table)):
                doc_id = i + 1
                tasks.append(self.process_question(
                    question_doc_id=doc_id,
                    overwrite=overwrite
                    ))

            # 并发执行所有任务
            await run_with_semaphore(tasks, max_concurrency,
                                               desc='[ScorePointsProcessor] Generating score points')
