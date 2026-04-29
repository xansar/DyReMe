from typing import List, Dict
from tqdm.asyncio import tqdm_asyncio
import asyncio
from utils.model import APIServer, VLLM, OpenAIModel, DeepSeekModel
import pandas as pd

import os
from copy import deepcopy

from .eval_prompts import INFER_SYSTEM_PROMPT, INFER_PROMPT
from utils import run_with_semaphore

from aiotinydb import AIOTinyDB
from tinydb import Query

import json
import random
random.seed(42)


class Inferencer:
    def __init__(
            self,
            model_name: str,
            api_server: APIServer,
            data_path: str,
            save_dir: str,
            # base_url: str = "http://localhost:8000/v1",
            # api_key: str="EMPTY",
            # sample_num: int = 1,
            dataset_num = None,
            temperature: float = 0.,
            max_tokens: int = 2048,
            # system_prompt: str="你是一个人工智能医学助手, 你的所有回答需要基于中文, 符合中文医学表述习惯.",
            # prompt: str="请回答以下问题：\n{question}"
        ) -> None:
        self.model_name = model_name
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_path = os.path.join(self.save_dir, f"{model_name.split('/')[-1]}.json")

        # self.sample_num = sample_num
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.system_prompt = system_prompt
        # self.prompt = prompt

        self.call_api = api_server.call_api

        self.dataset_num = dataset_num

        self._load_dataset()
        # self._init_caller(base_url, api_key)


        # caller = Caller()
        # self.call_api = caller.call_api

    def _load_dataset(self):
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        if self.dataset_num:
            random.shuffle(data)
            self.data = data[:self.dataset_num]
        else:
            self.data = data

    async def process_sample(self, sample: Dict, overwrite: bool = False):
        question_id = sample['question_id']
        if overwrite or not self.predictions_table.get(self.QueryModel.question_id == question_id):
            question = sample['question']

            system_prompt = INFER_SYSTEM_PROMPT
            prompt = INFER_PROMPT.format(
                question=question
            )
            
            result = await self.call_api(
                model=self.model_name,
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            if result:
                await asyncio.sleep(0.)
                # 保存结果
                prediction = deepcopy(sample)
                prediction['prediction'] = result
                prediction['prediction_id'] = question_id + f'_{self.model_name.split("/")[-1]}'
                self.predictions_table.upsert(prediction, self.QueryModel.question_id == question_id)
        else:
            await asyncio.sleep(0.)

    async def inference(self, overwrite: bool = False, max_concurrency: int = 8, position=0):
        # async with AIOTinyDB(self.data_path, indent=4, separators=(',', ': '), ensure_ascii=False) as db, \
        #     AIOTinyDB(self.save_path, indent=4, separators=(',', ': '), ensure_ascii=False) as predictions_db:
        
        async with AIOTinyDB(self.save_path, indent=4, separators=(',', ': '), ensure_ascii=False) as predictions_db:
            # self.questions_table = db.table('questions')
            self.predictions_table = predictions_db.table('predictions')
            if overwrite:
                self.predictions_table.truncate()

            tasks = []
            self.QueryModel = Query()
            for i, sample in enumerate(self.data):
                # if self.dataset_num:
                #     if i > self.dataset_num - 1:
                #         break
                tasks.append(self.process_sample(sample, overwrite))
            
            model = self.model_name.split('/')[-1].capitalize()

            await run_with_semaphore(tasks, max_concurrency, desc=f'[{model} Inference] Generating predictions', position=position)
            # await tqdm_asyncio.gather(
            #     *tasks, total=len(tasks),
            #     desc=f'[{model} Inference] Generating predictions')