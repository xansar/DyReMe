# import sys
import os
import json

# import traceback

# import pandas as pd
# from pydantic import BaseModel, Field
# from tqdm import tqdm
# from tqdm.asyncio import tqdm_asyncio

# from typing import List, Dict, Literal
# import json
import random
random.seed(42)
# import re
import asyncio
from tqdm.asyncio import tqdm_asyncio
# from itertools import chain

# from copy import deepcopy


from aiotinydb import AIOTinyDB
from tinydb import Query, TinyDB

import traceback

from .gen_prompts import TRAP_GENERATION_PROMPTS_DICT, MBTI_TYPES, EMOTIONS

from .raw_question_processor import RawQuestionProcessor
from .diagnosis_info_processor import DiagnosisInfoProcessor
from .misleading_question_generator import MisleadingQuestionGenerator
from .trap_question_processor import TrapQuestionProcessor
from .projected_gradient_descent_processor import ProjectedGradientDescentProcessor
from .score_points_processor import ScorePointsProcessor





class QuestionGenerator:
    def __init__(
            self,
            worker_model: str = 'gpt-4o-2024-08-06',
            generator_model: str = 'claude-3-5-sonnet-20240620',
            web_model: str = 'XXX',
            judge_model_lst: list = ['XXX'],

            raw_data_path: str = 'data.csv',
            data_path: str = 'data.csv',
            personas_path: str = 'personas.csv',
            knowledge_dir: str = 'diseases.csv',
            # save_dir: str = 'synthesis_questions',

            overwrite: bool = False,
            type_random: bool = False,
            repeat_num: int = 1,
            engine_name: str = 'Local',
            max_results: int = 3,
            use_hyde: bool = True,
            use_reranker: bool = True,
            use_summary: bool = False,
            random_temp: float = 0.5,
            decision_temp: float = 0.,
            eta: float = 0.3,
            max_attempts: int = 5,
            max_concurrency: int = 8
        ):
        self.worker_model = worker_model
        self.generator_model = generator_model
        self.web_model = web_model
        self.judge_model_lst = judge_model_lst

        self.raw_data_path = raw_data_path
        self.data_path = data_path

        self.personas_path = personas_path
        self.knowledge_dir = knowledge_dir
        # self.save_dir = save_dir
        # os.makedirs(self.save_dir, exist_ok=True)

        self.overwrite = overwrite
        self.type_random = type_random
        self.repeat_num = repeat_num

        self.random_temp = random_temp
        self.decision_temp = decision_temp
        self.eta = eta

        self.max_attemtps = max_attempts

        self.max_concurrency = max_concurrency

        self.engine_name = engine_name
        self.max_results = max_results
        self.use_hyde = use_hyde
        self.use_reranker = use_reranker
        self.use_summary = use_summary

        self.QueryModel = Query()
    
    async def _select_random_conditions(self, overwrite: bool = False, type_random: bool = True, knowledge_path: str = None):
        # 设置好问题的id，随机的干扰诊断、随机的陷阱以及随机的人格
        async with AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db, \
            AIOTinyDB(knowledge_path, indent=4, separators=(',', ': '), ensure_ascii=False) as knowledge_db, \
            AIOTinyDB(self.personas_path, indent=4, separators=(',', ': '), ensure_ascii=False) as personas_db:
            self.similar_diagnoses_table = knowledge_db.table('similar_diagnoses')

            self.personas_table = personas_db.table('personas')
            self.random_conditions_table = db.table('random_conditions')
            self.patients_table = db.table('patients')

            if overwrite:
                self.random_conditions_table.truncate()

            personas_num = len(self.personas_table)

            for patient in tqdm_asyncio(self.patients_table, desc='[QuestionGenerator] Selecting random conditions', ncols=100):
                patient_id = patient['patient_id']
                diagnosis = patient['diagnosis']
                symptoms = patient['symptoms']
                for i in range(self.repeat_num):

                    if not overwrite:
                        if type_random:
                            if self.repeat_num == 1:
                                if self.random_conditions_table.contains(self.QueryModel.patient_id == patient_id):
                                    continue
                            else:
                                cur_results = self.random_conditions_table.search(self.QueryModel.patient_id == patient_id)
                                if len(cur_results) >= self.repeat_num:
                                    continue


    
                    # 设定陷阱类型
                    raw_trap_types_lst = list(TRAP_GENERATION_PROMPTS_DICT.keys())
                    if self.type_random:
                        trap_type_lst = random.choices(raw_trap_types_lst, k=1)
                    else:
                        trap_type_lst = raw_trap_types_lst
                    
                    for trap_type in trap_type_lst:
                        if self.repeat_num == 1:
                            question_id = f'{patient_id}_{trap_type}'
                        else:
                            question_id = f'{patient_id}_{i}_{trap_type}'
                        # 这里的逻辑应该是怎样的？
                        # 考虑两种情况，如果type_random为True，那么每个病人的每个陷阱类型只会生成一次
                        # 因此只要有一个陷阱类型生成了，那么就不会再生成了
                        # 如果为False，那么每个病人的每个陷阱类型都会生成
                        # 因此只要没有生成，就会生成

                        if not overwrite:
                            if type_random:
                                if self.repeat_num == 1:
                                    if self.random_conditions_table.contains(self.QueryModel.patient_id == patient_id):
                                        continue
                                else:
                                    if self.random_conditions_table.contains(self.QueryModel.question_id == question_id):
                                        continue
                            else:
                                if self.random_conditions_table.contains(self.QueryModel.question_id == question_id):
                                    continue
                        else:
                            pass



                        # # 如果已经存在，跳过
                        # if not overwrite and self.random_conditions_table.contains(self.QueryModel.question_id == question_id):
                        #     continue
                        # 设定干扰诊断
                        similar_diagnoses = self.similar_diagnoses_table.get(
                            self.QueryModel.diagnosis_name == diagnosis)
                        if similar_diagnoses:
                            # symptoms = similar_diagnoses['symptoms']
                            refer_diagnosis_symptoms = similar_diagnoses['refer_diagnosis_symptoms']
                            similar_diagnoses = similar_diagnoses['similar_diagnoses']
                            distractor_diagnosis_info = random.choice(similar_diagnoses)
                        else:
                            raise ValueError(f'No similar diagnoses found for {diagnosis}')
                        
                        # 设定人格
                        ## 随机选取人物身份和行为
                        # import pdb; pdb.set_trace()
                        random_persona_id = f'persona_{random.randint(0, personas_num - 1)}'
                        Persona = Query()
                        persona = self.personas_table.get(Persona.persona_id == random_persona_id)
                        patient_desc = persona['persona']
                        patient_style = persona['persona_style']

                        # # 选取性格和情绪
                        # mbti_type = random.choice(MBTI_TYPES)
                        # emotion = random.choice(EMOTIONS)


                        # 从原始症状中随机选择一个用于生成误导性问题的症状
                        selected_symptom = random.choice(symptoms) if symptoms else None

                        # 保存随机条件
                        random_conditions = {
                            'patient_id': patient_id,
                            'question_id': question_id,
                            'diagnosis': diagnosis,
                            'symptoms': symptoms,
                            'selected_symptom': selected_symptom,  # 新增：记录选择的症状
                            'refer_diagnosis_symptoms': refer_diagnosis_symptoms,
                            'distractor_diagnosis_info': distractor_diagnosis_info,
                            'trap_type': trap_type,
                            'patient_desc': patient_desc,
                            'patient_style': patient_style,
                            # 'mbti_type': mbti_type,
                            # 'emotion': emotion
                        }
                        self.random_conditions_table.upsert(
                            random_conditions, self.QueryModel.question_id == question_id)
    
    async def _merge_misleading_questions(self, overwrite: bool = False, knowledge_path: str = None):
        async with AIOTinyDB(knowledge_path, indent=4, separators=(',', ': '), ensure_ascii=False) as knowledge_db, \
            AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db:
            self.misleading_knowledge_table = knowledge_db.table("misleading_knowledge")
            random_conditions_table = db.table('random_conditions')

            for condition in tqdm_asyncio(random_conditions_table, desc='[QuestionGenerator] Merging misleading questions', ncols=100):
                selected_symptom = condition.get('selected_symptom')  # 获取已选择的症状
                # 如果condition中已经有了误导问题，跳过
                if not overwrite and 'misleading_knowledge' in condition:
                    continue
                
                if not selected_symptom:
                    print(f'No selected symptom found for question: {condition.get("question_id", "unknown")}')
                    continue
                
                # 根据选择的症状查找对应的误导性知识
                misleading_record = self.misleading_knowledge_table.get(self.QueryModel.symptom == selected_symptom)
                # import pdb; pdb.set_trace()
                if not misleading_record or 'statement_pairs' not in misleading_record:
                    print(f'No misleading knowledge found for selected symptom: {selected_symptom}')
                    continue

                # 随机选取一个问题
                misleading_statement_pair = random.choice(misleading_record['statement_pairs'])
                condition['misleading_knowledge'] = misleading_statement_pair

                # 更新随机条件
                random_conditions_table.update(condition, self.QueryModel.question_id == condition['question_id'])

                

    async def _get_diagnosis_lst(self):
        async with AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db:
            patients_table = db.table('patients')

            diagnoses_lst = []
            # Patient = Query()
            for patient in patients_table:
                diagnosis = patient['diagnosis']
                diagnoses_lst.append(diagnosis)

        return diagnoses_lst
    
    async def _get_symptoms_lst(self):
        async with AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db:
            patients_table = db.table('patients')

            symptoms_lst = []
            # Patient = Query()
            for patient in patients_table:
                symptoms = patient['symptoms']
                symptoms_lst.append(symptoms)

        return symptoms_lst
    
    async def _get_selected_symptoms_lst(self):
        """获取每个问题已经选择的症状"""
        async with AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db:
            random_conditions_table = db.table('random_conditions')

            selected_symptoms_lst = []
            for condition in tqdm_asyncio(random_conditions_table, ncols=100, desc='[QuestionGenerator] Getting selected symptoms'):
                selected_symptom = condition.get('selected_symptom')
                if selected_symptom:  # 确保选择的症状存在
                    selected_symptoms_lst.append(selected_symptom)
            
        return selected_symptoms_lst

    
    async def save_generated_questions(self, output_filename: str = 'generated_questions.json'):
        """
        将生成的问题保存到JSON文件中
        
        Args:
            output_filename: 输出文件名，默认为'generated_questions.json'
        """
        async with AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db:
            questions_table = db.table('questions')
            questions_lst = questions_table.all()
            # 如果有score_points table
            score_points_flag = False
            if 'score_points' in db.tables():
                score_points_flag = True
                score_points_table = db.table('score_points')

            # 提取需要的字段
            extracted_questions = []
            for question in questions_lst:
                # import pdb; pdb.set_trace()
                if score_points_flag:
                    score_points = score_points_table.get(self.QueryModel.question_id == question['question_id'])
                    extracted_questions.append({
                        'question_id': question['question_id'],
                        'patient_id': question['patient_id'],
                        'diagnosis': question['refer_diagnosis'],
                        'question': question['question'],
                        'score_points': score_points["score_points"],
                        'correct_knowledge': question["intermediate_results"]["misleading_knowledge"]['correct_statement'],
                        'misleading_knowledge': question['intermediate_results']["misleading_knowledge"]['incorrect_statement']
                    })
                else:
                    extracted_questions.append({
                        'question_id': question['question_id'],
                        'patient_id': question['patient_id'],
                        'diagnosis': question['refer_diagnosis'],
                        'question': question['question'],
                        'correct_knowledge': question['intermediate_results']["misleading_knowledge"]['correct_statement'],
                        'misleading_knowledge': question['intermediate_results']["misleading_knowledge"]['incorrect_statement']
                    })

            
            # 保存到文件
            output_path = os.path.join(os.path.dirname(self.data_path), output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_questions, f, indent=4, ensure_ascii=False)
                
            print(f"Generated questions saved to: {output_path}")
            return output_path

    def generate(self, ablation_type: str = None):
        # self._load_data()

        # 生成原始问题
        raw_questions_synthesiser = RawQuestionProcessor(
            raw_data_path=self.raw_data_path,
            data_path=self.data_path,
            worker_model=self.worker_model,
            random_temp=self.random_temp
        )
        asyncio.run(raw_questions_synthesiser.run(overwrite=self.overwrite, max_concurrency=self.max_concurrency))
        # return 
        # diagnosis_lst, patient_id_lst = raw_questions_synthesiser.get_diagnosis_lst()
        # dedup_diagnosis_lst = list(set(diagnosis_lst))

        diagnosis_lst = asyncio.run(self._get_diagnosis_lst())
        dedup_diagnosis_lst = list(set(diagnosis_lst))


        # 构建相似诊断
        diagnosis_info_processor = DiagnosisInfoProcessor(
            worker_model=self.worker_model,
            web_model=self.web_model,
            random_temp=self.random_temp,
            knowledge_path=os.path.join(self.knowledge_dir, 'similar_diagnoses.json'),
        )
        # diagnosis_info_processor = DiagnosisInfoProcessor(
        #     worker_model=self.worker_model,
        #     random_temp=self.random_temp,
        #     knowledge_path=self.knowledge_path,
        #     engine_name=self.engine_name,
        #     max_results=self.max_results,
        #     use_hyde=self.use_hyde,
        #     use_reranker=self.use_reranker,
        #     use_summary=self.use_summary
        # )
        asyncio.run(diagnosis_info_processor.run(dedup_diagnosis_lst, overwrite=self.overwrite, max_concurrency=self.max_concurrency))

        # 设置随机条件
        asyncio.run(self._select_random_conditions(
            overwrite=self.overwrite, type_random=self.type_random,
            knowledge_path=os.path.join(self.knowledge_dir, 'similar_diagnoses.json')))

        # 为每个问题选择一个症状，然后去重
        selected_symptoms_lst = asyncio.run(self._get_selected_symptoms_lst())
        dedup_selected_symptoms_lst = list(set(selected_symptoms_lst))
        # 将单个症状包装成列表，因为误导性问题生成器期望症状列表的列表
        dedup_selected_symptoms_lst = [symptom for symptom in dedup_selected_symptoms_lst]

        # # 抽取相似诊断，并检查是否满足症状区分条件
        # similar_diagnoses_selector = SimilarDiagnosesSelector(
        #     worker_model=self.worker_model,
        #     random_temp=self.decision_temp,
        #     knowledge_path=os.path.join(self.knowledge_dir, 'similar_diagnoses.json'),
        # )
        # symptoms_lst = asyncio.run(self._get_symptoms_lst())
        # similar_diagnoses_lst = asyncio.run(similar_diagnoses_selector.run(diagnosis_lst, symptoms_lst, overwrite=self.overwrite, max_concurrency=self.max_concurrency))


        # 生成误导问题
        misleading_questions_generator = MisleadingQuestionGenerator(
            generator_model=self.generator_model, 
            web_model=self.web_model,
            judge_model_lst=self.judge_model_lst,
            decision_temp=self.decision_temp,
            random_temp=self.random_temp,
            data_path=self.data_path,
            knowledge_path=os.path.join(self.knowledge_dir, 'misleading_knowledge.json'),
            max_retries=self.max_attemtps,
            qa_triple_pairs_num_threshold=3,
            qa_triple_pairs_num=10,
        )
        asyncio.run(misleading_questions_generator.run(
            dedup_selected_symptoms_lst, 
            overwrite=self.overwrite, max_concurrency=self.max_concurrency))
        
        # 将误导问题合并到随机条件中
        asyncio.run(self._merge_misleading_questions(
            overwrite=self.overwrite, knowledge_path=os.path.join(self.knowledge_dir, 'misleading_knowledge.json')))
        
        # return 
        # distractor_diagnosis_misleading_questions_lst = \
        #     misleading_questions_generator.get_misleading_questions_lst(raw_distractor_diagnosis_lst)
        
        # # 合并干扰诊断和误导问题
        # distractor_info_lst = []
        # for diagnoses, misleading_questions in zip(raw_distractor_diagnosis_lst, distractor_diagnosis_misleading_questions_lst):
        #     cur_distractor_info_lst = []
        #     for diagnosis, misleading_question in zip(diagnoses, misleading_questions):
        #         cur_distractor_info_lst.append({
        #             'name': diagnosis['name'],
        #             'symptoms': diagnosis['symptoms'],
        #             'difference': diagnosis['difference'],
        #             'misleading_question': misleading_question
        #         })
        #     distractor_info_lst.append(cur_distractor_info_lst)

        # 生成陷阱问题
        trap_questions_processor = TrapQuestionProcessor(
            generator_model=self.generator_model,
            random_temp=self.random_temp,
            data_path=self.data_path,
            # knowledge_path=self.knowledge_dir,
            # personas_path=self.personas_path,
            type_random=self.type_random,
        )

        asyncio.run(trap_questions_processor.run(
            overwrite=self.overwrite, max_concurrency=self.max_concurrency, ablation_type=ablation_type))

        # return 
        # 验证和润色
        verify_and_refine_processor = ProjectedGradientDescentProcessor(
            data_path=self.data_path,
            generator_model=self.generator_model,
            eta=self.eta,
            # random_temp=self.random_temp,
            # decision_temp=self.decision_temp,
            max_iterations=5,
        )
        asyncio.run(verify_and_refine_processor.run(overwrite=self.overwrite, max_concurrency=self.max_concurrency))

        # # 生成得分点
        score_points_processor = ScorePointsProcessor(
            data_path=self.data_path,
            knowledge_path=os.path.join(self.knowledge_dir, 'score_points.json'),
            generator_model=self.generator_model,
            web_model=self.web_model,
            decision_temp=self.decision_temp,
        )
        asyncio.run(score_points_processor.run(overwrite=self.overwrite, max_concurrency=self.max_concurrency))

        # 生成结束后，将questions和对应的diagnosis，以及question_id,patient_id提取出来
        # 作为一个json文件保存
        output_filename = 'questions.json'
        asyncio.run(self.save_generated_questions(output_filename=output_filename))

        # 生成结束，print
        print('All questions generated successfully!')


