
import random
random.seed(42)
from typing import Dict, List

from tqdm.asyncio import tqdm_asyncio


from aiotinydb import AIOTinyDB
from tinydb import Query
from pydantic import BaseModel
import traceback

from .gen_prompts import (
    TRAP_PROMPTS, TRAP_GENERATION_PROMPTS_DICT,
    TRAP_PROMPTS_WO_DISTRACTOR, TRAP_PROMPTS_WO_PERSONA
    )

from utils import ModelCaller, run_with_semaphore




class TrapQuestionProcessor:
    def __init__(self, generator_model, 
    random_temp: float, data_path: str, type_random: bool = False):
        """
        初始化 TrapQuestionProcessor 类
        :param data: 数据源
        :param generator_model: 用于生成问题的模型
        :param random_temp: 随机温度值，用于控制生成的多样性
        :param personas: 人物信息数据
        :param db_path: 数据库路径
        """
        self.generator_model = generator_model
        self.random_temp = random_temp

        self.data_path = data_path
        # self.personas_path = personas_path
        # self.knowledge_path = knowledge_path

        self.type_random = type_random

        self._init_caller()

        self.QueryModel = Query()

    def _init_caller(self):
        caller = ModelCaller()
        self.call_api_json = caller.call_api_json
        self.call_api = caller.call_api

    async def _run_step(
            self, system_prompt: str, prompt_template: str, json_schema: BaseModel, **kwargs):
        """
        通用方法：执行单个步骤的任务
        :param system_prompt: 系统提示词
        :param prompt: 用户提示词
        :param output_model: 预期输出的 Pydantic 模型类型
        :return: 输出模型实例
        """

        prompt = prompt_template.format(**kwargs)

        response = await self.call_api_json(
            model=self.generator_model,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=self.random_temp,
            json_schema=json_schema,  # 自动生成 JSON schema
        )
        return response  # 使用 Pydantic 模型解析响应

    class Step1Output(BaseModel):
        TrapQuestion: str

    class Step2Output(BaseModel):
        MisleadingQuestion: str

    class Step3Output(BaseModel):
        PolishedPatientQuestion: str

    async def process_trap_wo_distractor(
            self, patient: Dict, random_condition: Dict, overwrite: bool = False):
        try:
            patient_id = patient["patient_id"]
            question_id = random_condition['question_id']
            
            raw_question = patient['raw_question']['description'] + '\n' + patient['raw_question']['question']
            refer_diagnosis = random_condition['diagnosis']
            org_symptoms_lst = random_condition['symptoms']
            # refer_diagnosis_symptoms = random_condition['symptoms']
            trap_type = random_condition['trap_type']
                
            # question_id = f'{patient_id}_{trap_type}'
            if not overwrite and self.trap_questions_table.contains(self.QueryModel.question_id == question_id):
                return
            
            trap_type_name = TRAP_GENERATION_PROMPTS_DICT[trap_type]['trap_type']
            trap_desc = TRAP_GENERATION_PROMPTS_DICT[trap_type]['description']
            trap_task_description = TRAP_GENERATION_PROMPTS_DICT[trap_type]['task_description']



            # 选取反事实诊断
            distractor_diagnosis = random_condition['distractor_diagnosis_info']['name']
            # distractor_diagnosis_symptoms = random_condition['distractor_diagnosis_info']['symptoms']
            # diagnosis_difference = random_condition['distractor_diagnosis_info']['difference']

            # 随机选取人物身份和行为
            patient_desc = random_condition['patient_desc']
            patient_style = random_condition['patient_style']
            # patient_behaviours = random_condition['patient_behaviours']

            # # 选取性格和情绪
            # mbti_type = random_condition['mbti_type']
            # emotion = random_condition['emotion']

            # with tqdm_asyncio(total=3, desc=f"[Traps] Processing {question_id}'s trap", leave=False) as pbar:
            # pbar.update(1)


            # Step 2 将误导性知识融入问题
            import pdb; pdb.set_trace()
            step_2_result = await self._run_step(
                TRAP_PROMPTS['step_2']['system_prompt'],
                TRAP_PROMPTS['step_2']['prompt_template'],
                self.Step2Output,
                trap_question=raw_question,
                misleading_knowledge=random_condition['misleading_knowledge'],
            )
            # pbar.update(1)


            # Step 3 基于persona调整问题表述风格
            step_3_result = await self._run_step(
                TRAP_PROMPTS['step_3']['system_prompt'],
                TRAP_PROMPTS['step_3']['prompt_template'],
                self.Step3Output,
                raw_question=step_2_result['MisleadingQuestion'],
                patient_style=patient_style,
            )
            # pbar.update(1)



            trap_question = {
                'question_id': question_id,
                'patient_id': patient_id,
                'type': trap_type,
                'question': step_3_result['PolishedPatientQuestion'],
                'refer_diagnosis': refer_diagnosis,
                "org_symptoms_lst": org_symptoms_lst,
                'distractor_diagnosis': distractor_diagnosis,
                "selected_symptoms": random_condition['selected_symptom'],
                "patient_desc": patient_desc,
                "patient_style": patient_style,
                "trap_info": {
                    "trap_type": trap_type_name,
                    "trap_desc": trap_desc,
                    "trap_task_description": trap_task_description,
                },
                'intermediate_results': {
                    # 'trap_question': step_1_result['TrapQuestion'],
                    'misleading_knowledge': random_condition['misleading_knowledge'],
                    'misleading_question': step_2_result['MisleadingQuestion'],
                    'polished_patient_question': step_3_result['PolishedPatientQuestion'],
                }
            }

            
            self.trap_questions_table.upsert(trap_question, self.QueryModel.question_id == trap_question['question_id'])
        
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            traceback.print_exc()

    async def process_trap_wo_persona(
            self, patient: Dict, random_condition: Dict, overwrite: bool = False):
        try:
            patient_id = patient["patient_id"]
            question_id = random_condition['question_id']

            raw_question = patient['raw_question']['description'] + '\n' + patient['raw_question']['question']
            refer_diagnosis = random_condition['diagnosis']
            org_symptoms_lst = random_condition['symptoms']
            trap_type = random_condition['trap_type']
                
            if not overwrite and self.trap_questions_table.contains(self.QueryModel.question_id == question_id):
                return

            trap_type_name = TRAP_GENERATION_PROMPTS_DICT[trap_type]['trap_type']
            trap_desc = TRAP_GENERATION_PROMPTS_DICT[trap_type]['description']
            trap_task_description = TRAP_GENERATION_PROMPTS_DICT[trap_type]['task_description']

            # 选取反事实诊断
            distractor_diagnosis = random_condition['distractor_diagnosis_info']['name']

            # Step 1 将问题改写为陷阱问题
            step_1_result = await self._run_step(
                TRAP_PROMPTS['step_1']['system_prompt'], 
                TRAP_PROMPTS['step_1']['prompt_template'],
                self.Step1Output,
                raw_question=raw_question,
                org_symptoms_lst=org_symptoms_lst,
                refer_diagnosis=refer_diagnosis,
                trap_type_name=trap_type_name,
                trap_desc=trap_desc,
                trap_task_description=trap_task_description,
                distractor_diagnosis=distractor_diagnosis,
            ) 

            # Step 2 将误导性知识融入问题
            step_2_result = await self._run_step(
                TRAP_PROMPTS['step_2']['system_prompt'],
                TRAP_PROMPTS['step_2']['prompt_template'],
                self.Step2Output,
                trap_question=step_1_result['TrapQuestion'],
                misleading_knowledge=random_condition['misleading_knowledge'],
            )

            trap_question = {
                'question_id': question_id,
                'patient_id': patient_id,
                'type': trap_type,
                'question': step_2_result['MisleadingQuestion'],
                'refer_diagnosis': refer_diagnosis,
                "org_symptoms_lst": org_symptoms_lst,
                'distractor_diagnosis': distractor_diagnosis,
                "selected_symptoms": random_condition['selected_symptom'],
                "trap_info": {
                    "trap_type": trap_type_name,
                    "trap_desc": trap_desc,
                    "trap_task_description": trap_task_description,
                },
                'intermediate_results': {
                    'trap_question': step_1_result['TrapQuestion'],
                    'misleading_knowledge': random_condition['misleading_knowledge'],
                    'misleading_question': step_2_result['MisleadingQuestion'],
                }
            }

            self.trap_questions_table.upsert(trap_question, self.QueryModel.question_id == trap_question['question_id'])

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            traceback.print_exc()
        
    async def process_trap(
            self, patient: Dict, random_condition: Dict,
            overwrite: bool = False):
        try:
            patient_id = patient["patient_id"]
            question_id = random_condition['question_id']
            
            raw_question = patient['raw_question']['description'] + '\n' + patient['raw_question']['question']
            refer_diagnosis = random_condition['diagnosis']
            org_symptoms_lst = random_condition['symptoms']
            # refer_diagnosis_symptoms = random_condition['symptoms']
            trap_type = random_condition['trap_type']
                
            # question_id = f'{patient_id}_{trap_type}'
            if not overwrite and self.trap_questions_table.contains(self.QueryModel.question_id == question_id):
                return
            
            trap_type_name = TRAP_GENERATION_PROMPTS_DICT[trap_type]['trap_type']
            trap_desc = TRAP_GENERATION_PROMPTS_DICT[trap_type]['description']
            trap_task_description = TRAP_GENERATION_PROMPTS_DICT[trap_type]['task_description']



            # 选取反事实诊断
            distractor_diagnosis = random_condition['distractor_diagnosis_info']['name']
            # distractor_diagnosis_symptoms = random_condition['distractor_diagnosis_info']['symptoms']
            # diagnosis_difference = random_condition['distractor_diagnosis_info']['difference']

            # 随机选取人物身份和行为
            patient_desc = random_condition['patient_desc']
            patient_style = random_condition['patient_style']
            # patient_behaviours = random_condition['patient_behaviours']

            # # 选取性格和情绪
            # mbti_type = random_condition['mbti_type']
            # emotion = random_condition['emotion']

            # with tqdm_asyncio(total=3, desc=f"[Traps] Processing {question_id}'s trap", leave=False) as pbar:

            # Step 1 将问题改写为陷阱问题
            step_1_result = await self._run_step(
                TRAP_PROMPTS['step_1']['system_prompt'], 
                TRAP_PROMPTS['step_1']['prompt_template'],
                self.Step1Output,
                raw_question=raw_question,
                org_symptoms_lst=org_symptoms_lst,
                refer_diagnosis=refer_diagnosis,
                trap_type_name=trap_type_name,
                trap_desc=trap_desc,
                trap_task_description=trap_task_description,
                distractor_diagnosis=distractor_diagnosis,
                # differential_symptoms=step_3_result['DifferentialSymptoms'],
            ) 
            # pbar.update(1)


            # Step 2 将误导性知识融入问题
            step_2_result = await self._run_step(
                TRAP_PROMPTS['step_2']['system_prompt'],
                TRAP_PROMPTS['step_2']['prompt_template'],
                self.Step2Output,
                trap_question=step_1_result['TrapQuestion'],
                misleading_knowledge=random_condition['misleading_knowledge'],
            )
            # pbar.update(1)


            # Step 3 基于persona调整问题表述风格
            step_3_result = await self._run_step(
                TRAP_PROMPTS['step_3']['system_prompt'],
                TRAP_PROMPTS['step_3']['prompt_template'],
                self.Step3Output,
                raw_question=step_2_result['MisleadingQuestion'],
                patient_style=patient_style,
            )
            # pbar.update(1)

            trap_question = {
                'question_id': question_id,
                'patient_id': patient_id,
                'type': trap_type,
                'question': step_3_result['PolishedPatientQuestion'],
                'refer_diagnosis': refer_diagnosis,
                "org_symptoms_lst": org_symptoms_lst,
                'distractor_diagnosis': distractor_diagnosis,
                "selected_symptoms": random_condition['selected_symptom'],
                "patient_desc": patient_desc,
                "patient_style": patient_style,
                "trap_info": {
                    "trap_type": trap_type_name,
                    "trap_desc": trap_desc,
                    "trap_task_description": trap_task_description,
                },
                'intermediate_results': {
                    'trap_question': step_1_result['TrapQuestion'],
                    'misleading_knowledge': random_condition['misleading_knowledge'],
                    'misleading_question': step_2_result['MisleadingQuestion'],
                    'polished_patient_question': step_3_result['PolishedPatientQuestion'],
                }
            }

            
            self.trap_questions_table.upsert(trap_question, self.QueryModel.question_id == trap_question['question_id'])
        
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            traceback.print_exc()


    async def run(self, overwrite: bool = False, max_concurrency=8, ablation_type: str = None):
        """
        生成所有数据实例的陷阱问题
        """
        async with AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db:
            self.patients_table = db.table('patients')
            if ablation_type == 'wo_distractor':
                self.trap_questions_table = db.table('questions_wo_distractor')
            elif ablation_type == 'wo_persona':
                self.trap_questions_table = db.table('questions_wo_persona')
            else:
                self.trap_questions_table = db.table('questions')
            self.random_conditions_table = db.table('random_conditions')
            if overwrite:
                self.trap_questions_table.truncate()

            # self.similar_diagnoses_table = knowledge_db.table('similar_diagnoses')

            # self.personas_table = personas_db.table('personas')

            # tasks = []
            # for id in range(len(self.patients_table)):
            #     patient_id = f'patient_{id}'

            #     patient = self.patients_table.get(self.QueryModel.patient_id == patient_id)
            #     tasks.append(self.process_trap(
            #         patient, 
            #         overwrite=overwrite))

            tasks = []
            for random_condition in self.random_conditions_table:
                # print(random_condition['question_id'])
                patient_id = random_condition['patient_id']
                patient = self.patients_table.get(self.QueryModel.patient_id == patient_id)
                if ablation_type == 'wo_distractor':
                    tasks.append(self.process_trap_wo_distractor(
                        patient,
                        random_condition, 
                        overwrite=overwrite))
                elif ablation_type == 'wo_persona':
                    tasks.append(self.process_trap_wo_persona(
                        patient,
                        random_condition, 
                        overwrite=overwrite))
                else:
                    tasks.append(self.process_trap(
                        patient,
                        random_condition, 
                        overwrite=overwrite))

            # 并发执行所有任务
            await run_with_semaphore(tasks, max_concurrency,
                                               desc='[TrapQuestionProcessor] Generating trap questions')
            # await tqdm_asyncio.gather(*tasks, total=len(tasks),
            #                           desc='[TrapQuestionProcessor] Generating trap questions')
        