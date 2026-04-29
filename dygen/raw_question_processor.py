from copy import deepcopy

from aiotinydb import AIOTinyDB
from tinydb import Query,TinyDB
from pydantic import BaseModel
import traceback

from .gen_prompts import SYNTHESIS_RAW_QUESTION_PROMPT, SYNTHESIS_RAW_QUESTION_SYSTEM_PROMPT

from utils import ModelCaller, run_with_semaphore
class RawQuestionProcessor:
    def __init__(self, raw_data_path: str, data_path: str, worker_model, random_temp: float):
        """
        初始化 RawQuestionProcessor 类
        :param data_path: 数据库路径
        :param worker_model: 用于生成问题的模型
        :param random_temp: 随机温度值，用于控制生成的多样性
        """
        self.raw_data_path = raw_data_path
        self.data_path = data_path

        self.worker_model = worker_model
        self.random_temp = random_temp
        self._init_caller()

        self.QueryModel = Query()

    class SynthesisQuery(BaseModel):
        description: str
        question: str

    def _init_caller(self):
        caller = ModelCaller()
        self.call_api_json = caller.call_api_json

    async def process_patient(self, patient, overwrite: bool = False):
        """
        处理单个患者的 raw_question 数据，并更新数据库
        :param patients_table: TinyDB 表对象
        :param doc_id: 患者文档 ID
        """


        patient_id = patient['patient_id']
        save_patient = self.save_table.get(self.QueryModel.patient_id == patient_id)
        
        # 如果患者已经有 raw_question，跳过
        if not overwrite and save_patient and save_patient.get('raw_question', None):
            return
    
        # 如果patient内有raw_question，跳过
        if patient.get('raw_question', None):
            # 执行数据库更新
            update_patient = deepcopy(patient)
            self.save_table.upsert(
                update_patient, self.QueryModel.patient_id == patient_id)
            return
        # if not overwrite and patient.get('raw_question', None):
        #     return
        
        # gender = patient["sex"]
        # age = patient["age"]
        # gender = '男' if gender == 'M' else '女'
        # age = f"{age}岁"
        symptoms = patient["symptoms"]
        if 'muzhi' in self.raw_data_path or '儿' in patient['diagnosis']:
            pronoun_tone = '患者是儿童，请以患者父母口吻描述病情。'
        else:
            pronoun_tone = '使用患者第一人称自述口吻描述病情。'

        # 拼接 prompt
        sys_prompt = SYNTHESIS_RAW_QUESTION_SYSTEM_PROMPT
        prompt = SYNTHESIS_RAW_QUESTION_PROMPT.format(
            symptoms=symptoms,
            pronoun_tone=pronoun_tone
        )

        try:
            # 调用 API 获取结果
            result = await self.call_api_json(
                model=self.worker_model,
                system_prompt=sys_prompt,
                prompt=prompt,
                temperature=self.random_temp,
                json_schema=self.SynthesisQuery,
            )

            # 执行数据库更新
            update_patient = deepcopy(patient)
            update_patient['raw_question'] = result
            self.save_table.upsert(
                update_patient, self.QueryModel.patient_id == patient_id)

        except Exception as e:
            print(f"Error processing patient_{id}: {e}")

    def get_diagnosis_lst(self):
        self._init_caller()
        with TinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db:
            patients_table = db.table('patients')

            diagnoses_lst = []
            patient_id_lst = []
            # Patient = Query()
            for patient in patients_table:
                patient_id = patient['patient_id']
                diagnosis = patient['diagnosis']
                diagnoses_lst.append(diagnosis)
                patient_id_lst.append(patient_id)

        return diagnoses_lst, patient_id_lst

    async def run(self, overwrite: bool = False, max_concurrency: int = 8):
        """
        处理所有患者的 raw_question 数据
        """
        async with AIOTinyDB(self.raw_data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db, \
            AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as save_db:
            self.patients_table = db.table('patients')
            self.save_table = save_db.table('patients')
            # print(f"Processing {len(self.patients_table)} patients")
            # print(f"Processing {len(self.save_table)} patients")
            tasks = []

            if overwrite:
                self.save_table.truncate()

            # 收集所有处理任务
            for patient in self.patients_table:
                # if self.save_table.contains(self.QueryModel.patient_id == patient['patient_id']) and not overwrite:
                #     continue
                tasks.append(self.process_patient(patient, overwrite=overwrite))

            # 并发执行所有任务
            await run_with_semaphore(tasks, max_concurrency,
                                               desc='[RawQuestionProcessor] Synthesizing raw questions')
            # await tqdm_asyncio.gather(*tasks, total=len(tasks), 
            #                           desc='[RawQuestionProcessor] Synthesizing raw questions')
