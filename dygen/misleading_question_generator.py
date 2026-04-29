from __future__ import annotations

import random
random.seed(42)
from typing import List, Optional, Tuple
import asyncio


from aiotinydb import AIOTinyDB
from tinydb import Query, TinyDB
from pydantic import BaseModel
import traceback
from tqdm.asyncio import tqdm_asyncio


from .gen_prompts import (
    DIAGNOSIS_INFO_GENERATE_SYSTEM_PROMPT, DIAGNOSIS_INFO_GENERATE_PROMPT,
    MISLEADING_QUESTIONS_GENERATE_SYSTEM_PROMPT, MISLEADING_QUESTIONS_GENERATE_PROMPT,
    MISLEADING_STATEMENTS_GENERATE_SYSTEM_PROMPT, MISLEADING_STATEMENTS_GENERATE_PROMPT
)

# from model import OpenAIModel, DeepSeekModel 
from utils import ModelCaller, run_with_semaphore

import logging
from dataclasses import dataclass
from typing import List, Dict

class MisleadingQuestionGenerator:
    def __init__(
            self, 
            generator_model, 
            web_model,
            judge_model_lst,
            decision_temp, 
            random_temp,
            data_path, 
            knowledge_path, 
            max_retries=5,
            qa_triple_pairs_num=10,
            qa_triple_pairs_num_threshold=5
        ):
        """
        初始化诊断信息处理器
        :param worker_model: 用于生成问题的模型
        :param random_temp: 随机温度值，用于控制生成的多样性
        :param knowledge_path: 疾病数据库路径
        """
        self.generator_model = generator_model
        self.web_model = web_model
        self.judge_model_lst = judge_model_lst

        self.decision_temp = decision_temp
        self.random_temp = random_temp
        self.knowledge_path = knowledge_path
        self.data_path = data_path

        self.max_retries = max_retries
        self.statement_pairs_num = qa_triple_pairs_num
        self.statement_pairs_num_threshold = qa_triple_pairs_num_threshold

        self._init_caller()

        self.QueryModel = Query()

    def _init_caller(self):
        caller = ModelCaller()
        self.call_api = caller.call_api
        self.call_api_json = caller.call_api_json
        self.call_api_web = caller.call_api_web
    

    class StatementPairList(BaseModel):
        class StatementPair(BaseModel):
            # question: str
            correct_statement: str
            incorrect_statement: str
        statement_pairs: List[StatementPair]

    async def _search_and_cache_symptom_info(self, symptom: str) -> str:
        """
        搜索并缓存症状相关信息。
        
        :param symptom: 症状描述
        :return: 搜索到的相关信息文本
        """
        # 先检查是否已经缓存了该症状的信息
        cached_info = self.symptom_info_table.get(self.QueryModel.symptom == symptom)
        if cached_info:
            return cached_info['info'], cached_info['references']
        
        # 构建搜索查询
        # search_query = f"{symptom} 诱因、用药、相关疾病、自我处理、检查、危险信号、病理机制、诊断要点、鉴别诊断、治疗建议、生活方式建议"
        
        # 使用web模型搜索信息
        search_result, references = await self.call_api_web(
            model=self.web_model,
            system_prompt="你是一个医学信息搜索助手。请搜索并提供与给定症状相关的准确医学信息。",
            prompt=f"请搜索关于'{symptom}'的详细医学信息，包括诱因、用药、相关疾病、自我处理、检查、危险信号、病理机制、诊断要点、鉴别诊断、治疗建议、生活方式建议等内容。",
            temperature=self.decision_temp,
            max_tokens=128,
        )
        
        # 将搜索结果缓存到表中
        self.symptom_info_table.upsert(
            {
                "symptom": symptom,
                "info": search_result,
                'references': references
            },
            self.QueryModel.symptom == symptom,
        )
        
        return search_result, references

    async def _generate_statement_pairs(self, symptom: str, n: int):
        """
        基于给定症状生成n个正确、错误陈述对。

        :param symptom: 症状描述
        :param n: 生成的陈述对数量
        :return: 正确、错误的陈述对的列表或None。
        """
        # 先搜索并缓存症状信息
        _, references = await self._search_and_cache_symptom_info(symptom)

        if not references:
            references = []
        references = references[:5]
        symptom_info = "\n\n".join([f"{ref['title']}\n{ref['summary']}" for ref in references])
        
        # 基于搜索到的信息生成陈述对
        system_prompt = MISLEADING_STATEMENTS_GENERATE_SYSTEM_PROMPT
        prompt = MISLEADING_STATEMENTS_GENERATE_PROMPT.format(
            symptom=symptom, 
            symptom_info=symptom_info,
            n=n
        )

        result = await self.call_api_json(
            model=self.generator_model,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=self.random_temp,
            json_schema=self.StatementPairList
        )

        statement_pairs = result["statement_pairs"]
        return statement_pairs



    async def _rationability_check(self, correct_statement: str, incorrect_statement: str) -> bool:
        # 利用联网大模型验证正确陈述是否合理，错误陈述是否不合理
        # 1. 验证正确陈述是否合理
        system_prompt = "请判断**陈述**是否是正确的。如果是，请回答'正确'；否则回答'错误'。禁止回复其他内容。"
        prompt = "**陈述:** {statement}\n\n**输出:**"

        # 验证正确
        correct_response, _ = await self.call_api_web(
            model=self.web_model,
            system_prompt=system_prompt,
            prompt=prompt.format(statement=correct_statement),
            temperature=self.decision_temp
        )
        if correct_response == "正确":
            distractor_response, _ = await self.call_api_web(
                model=self.web_model,
                system_prompt=system_prompt,
                prompt=prompt.format(statement=incorrect_statement),
                temperature=self.decision_temp
            )
            if distractor_response == "错误":
                return True
            
        return False
    

    async def _difficulty_check(self, incorrect_statement):
        # 通过是否有Judge model认同incorrect_statement来说明问题的挑战性
            tasks = []
            system_prompt = """你是一名人工智能助手。"""
            prompt = f"请判断下面的陈述是否正确：{incorrect_statement}\n如果正确，请回答'正确'；如果错误，请回答'错误'。禁止回复其他内容。"
            for model_info in self.judge_model_lst:
                tasks.append(self.call_api(
                    model=model_info['name'],
                    system_prompt=system_prompt,
                    prompt=prompt,
                    temperature=self.decision_temp,
                    provider_kwargs=model_info.get('provider_kwargs')
                ))
            
            results = await asyncio.gather(*tasks)
            
            # import pdb; pdb.set_trace()
            # 验证是否有模型回答正确，如果有则通过
            eval_results = []
            for r in results:
                # 判断模型的回答是否正确
                if "正确" in r.strip():
                    eval_results.append(True)
                else:
                    eval_results.append(False)
            
            # 至少有一个模型需要回答为正确，才认为这个问题有难度
            return any(eval_results), eval_results

        

    async def _check_statement(self, correct_statement: str, incorrect_statement: str) -> Tuple[bool, List[bool]]:
        # 验证问题是否合理
        rationality = await self._rationability_check(correct_statement, incorrect_statement)
        if not rationality:
            return False, "not rational"

        # # 验证问题的难度（使用问答式）
        # difficulty, model_perf_results = await self._difficulty_check(incorrect_statement)
        # if not difficulty:
        #     return False, "not difficult"

        return True, None
        # return True, model_perf_results


    async def process_symptom(
        self,
        symptom_key: str,
        symptom: str,
        overwrite: bool = False,
    ) -> None:
        """为单个症状组合生成误导性知识并落库。"""

        # ---------- 1. 提前返回 ----------
        if not overwrite and self._has_enough_pairs(symptom):
            print(f"{symptom} - 已有足够问答对，跳过")
            return

        # ---------- 2. 收集有效陈述对 ----------
        valid_pairs = await self._collect_valid_pairs(symptom)
        if not valid_pairs:
            print(f"{symptom} - 无法生成有效问答对")
            return

        # ---------- 3. 一次性写库 ----------
        await self._persist_pairs(symptom, valid_pairs)
        print(f"{symptom} - 写入 {len(valid_pairs)} 条误导性问答")


    # ---------------------------------------------------------------------
    # ↓↓↓ 以下辅助函数把原先的嵌套循环拆开，每个函数只做一件事 ↓↓↓
    # ---------------------------------------------------------------------

    def _has_enough_pairs(self, symptom: str) -> bool:
        row = self.misleading_knowledge_table.get(
            self.QueryModel.symptom == symptom
        )
        return bool(row and len(row.get("statement_pairs", [])) >= self.statement_pairs_num_threshold)


    async def _collect_valid_pairs(
        self, symptom: str
    ):
        """多轮生成 + 校验，直到达到阈值或用尽重试次数。"""

        valid = []
        for attempt in range(1, self.max_retries + 1):
            raw_pairs = await self._generate_statement_pairs(
                symptom, n=self.statement_pairs_num
            )
            if not raw_pairs:
                print(f"症状 {str(symptom)} - 第 {attempt} 次生成失败")
                continue

            # 并发校验
            check_results = await asyncio.gather(
                *[
                    self._check_statement(p["correct_statement"], p["incorrect_statement"])
                    # self._check_statement(p["question"], p["correct_statement"], p["incorrect_statement"])
                    for p in raw_pairs
                ]
            )
            # import pdb; pdb.set_trace()
            # 过滤通过的
            for rp, (ok, difficulty) in zip(raw_pairs, check_results):
                if ok:
                    valid.append(
                        {
                            # "question": rp["question"],
                            "correct_statement": rp["correct_statement"],
                            "incorrect_statement": rp["incorrect_statement"],
                            # "difficulty": {
                            #     m['name']: r for m, r in zip(self.judge_model_lst, difficulty)
                            # }
                        }
                    )

            if len(valid) >= self.statement_pairs_num_threshold:
                break
            
            # import pdb; pdb.set_trace()
            print(
                f"症状 {str(symptom)} - 第 {attempt} 次仅通过 {len(valid)}/{self.statement_pairs_num_threshold}，继续重试",
            )
        return valid


    async def _persist_pairs(self, symptom: str, pairs) -> None:
        """把新生成的问答对追加到数据库。"""

        row = self.misleading_knowledge_table.get(self.QueryModel.symptom == symptom)
        stored = row.get("statement_pairs", []) if row else []

        # dataclass → dict
        stored.extend(pairs)
        # import pdb; pdb.set_trace()
        self.misleading_knowledge_table.upsert(
            {"symptom": symptom, "statement_pairs": stored},
            self.QueryModel.symptom == symptom,
        )


    async def run(self, symptom_lst, overwrite, max_concurrency = 8):
        """
        生成与给定症状相关的误导性知识。

        :param symptoms_lst: 症状列表，每个元素包含症状组合。
        :return: 误导性知识和解释的元组或None。
        """
        from utils import run_with_semaphore
        
        async with AIOTinyDB(self.knowledge_path, indent=4, separators=(',', ': '), ensure_ascii=False) as knowledge_db:
            self.misleading_knowledge_table = knowledge_db.table("misleading_knowledge")
            self.symptom_info_table = knowledge_db.table("symptom_info")

            tasks = []
            for i, symptom in enumerate(symptom_lst):
                # 为症状组合生成一个唯一的键
                symptom_key = f"symptom_{i}_{hash(str(symptom)) % 100000}"
                tasks.append(self.process_symptom(symptom_key, symptom, overwrite))

            await run_with_semaphore(tasks, max_concurrency, desc='[MisleadingKnowledgeGenerator] Generating misleading knowledge')