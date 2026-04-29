
import asyncio
from copy import deepcopy
from typing import Literal, Dict, Any, List, Tuple

from aiotinydb import AIOTinyDB
from tinydb import Query
from pydantic import BaseModel, Field
import traceback
from tqdm.asyncio import tqdm_asyncio

from .gen_prompts import VERIFY_PROMPT, REFINE_PROMPT, REASON_PROMPT, VERIFY_SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT

from utils import ModelCaller, run_with_semaphore

class ProjectedGradientDescentProcessor:
    """
    基于投影梯度下降框架的诊断问题优化器
    
    实现论文中描述的约束优化问题：
    q_{t+1} = Π_C(q_t - η ∇L(q_t))
    
    其中：
    - q_t: 第t次迭代的候选问题
    - L(q_t): 代理损失函数，反映约束违反程度  
    - η: 步长（学习率）
    - Π_C: 投影到可行集C（临床有效问题集合）
    - C: 满足所有临床标准的问题集合
    """
    
    def __init__(self, data_path, generator_model, eta=0.3, max_iterations=5):
        """
        初始化投影梯度下降处理器
        
        Args:
            data_path: 数据文件路径
            generator_model: 生成模型
            eta: 步长参数（对应公式中的η），控制优化强度
                - 0.0-0.3: 保守优化，进行最小化精确修改
                - 0.3-0.7: 中等优化，进行适度的问题调整
                - 0.7-1.0: 激进优化，进行较大幅度的重构
            max_iterations: 最大迭代次数
        """
        self.data_path = data_path
        self.generator_model = generator_model
        self.eta = eta  # 步长参数
        self.max_iterations = max_iterations
        
        self._init_caller()
        self.QueryModel = Query()

    def _init_caller(self):
        """初始化模型调用器"""
        caller = ModelCaller()
        self.call_api_json = caller.call_api_json
        self.call_api = caller.call_api

    def _log_iteration(self, question_id: str, iteration: int, action_type: str, 
                      question: str, result: Any, reason: Any = None):
        """记录迭代过程"""
        log_entry = {
            "iteration": iteration,
            "action_type": action_type,
            "question": question,
            "result": result,
            "reason": reason,
        }
        
        existing_logs = self.logs_table.get(self.QueryModel.question_id == question_id)
        if not existing_logs:
            new_logs = [log_entry]
        else:
            new_logs = existing_logs['logs'] + [log_entry]
        
        self.logs_table.upsert({
            'question_id': question_id,
            'logs': new_logs
        }, self.QueryModel.question_id == question_id)

    class ValidationResult(BaseModel):
        """验证器V的输出结果"""
        class CriterionResult(BaseModel):
            assessment: str = Field(..., description='分析结果')
            verify_result: Literal["通过", "不通过"] = Field(..., description='是否满足约束')
            
        challenge: CriterionResult = Field(..., description='挑战性约束')
        rationality: CriterionResult = Field(..., description='合理性约束')
        trap_integrity: CriterionResult = Field(..., description='陷阱完整性约束')
        style_consistency: CriterionResult = Field(..., description='患者风格一致性约束')
        misleading_embedding: CriterionResult = Field(..., description='误导性知识嵌入约束')

    class RefinementResult(BaseModel):
        """优化器R的输出结果"""
        gradient_explanation: str = Field(..., description='梯度方向解释（如何调整以减少约束违反）')
        refined_question: str = Field(..., description='优化后的问题')

    async def validator(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
            验证器V：检查候选问题是否满足所有临床约束
            
            对应数学公式中的约束集合C的成员检查
            
            Returns:
                (is_in_feasible_set, constraint_violations)
            """
        prompt = VERIFY_PROMPT.format(
                question=sample['question'],
                refer_diagnosis=sample['refer_diagnosis'],
                org_symptoms_lst=sample['org_symptoms_lst'],
                distractor_diagnosis=sample['distractor_diagnosis'],
                selected_symptoms=sample['selected_symptoms'],
                patient_desc=sample['patient_desc'],
                patient_style=sample['patient_style'],
                misleading_knowledge=sample["intermediate_results"]['misleading_knowledge']['incorrect_statement'],
            )

        validation_result = await self.call_api_json(
            model=self.generator_model,
            system_prompt=VERIFY_SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.0,  # 确定性验证
            json_schema=self.ValidationResult
        )

        # 检查是否所有约束都满足（是否在可行集C中）
        is_in_feasible_set = all([
            criterion['verify_result'] == '通过' 
            for criterion in validation_result.values()
        ])

        return is_in_feasible_set, validation_result

    def _get_adaptive_eta(self, iteration: int, violation_count: int) -> float:
        """
        根据迭代次数和违反约束数量动态调整eta
        
        Args:
            iteration: 当前迭代次数
            violation_count: 违反约束的数量
            
        Returns:
            调整后的eta值
        """
        # 基础eta随迭代次数递增（后期迭代更激进）
        base_eta = self.eta * (1 + iteration * 0.2)
        
        # 根据违反约束数量调整（违反越多，调整越激进）
        violation_multiplier = 1 + violation_count * 0.1
        
        # 确保eta不超过1.0
        adaptive_eta = min(1.0, base_eta * violation_multiplier)
        
        return adaptive_eta

    async def refinement_operator(self, sample: Dict[str, Any], 
                                 constraint_violations: Dict[str, Any],
                                 iteration: int = 0) -> Dict[str, Any]:
        """
        优化器R：计算并应用"梯度"步骤
        
        对应数学公式中的 q_t - η∇L(q_t) 操作
        梯度方向指向减少约束违反的方向
        
        Args:
            sample: 当前样本
            constraint_violations: 约束违反信息
            iteration: 当前迭代次数
        """
        # 构造约束违反信息（相当于梯度信息）
        violation_feedback = {
            k: f"分析：{v['assessment']}\n结果：{v['verify_result']}"
            for k, v in constraint_violations.items()
        }
        
        reason_text = REASON_PROMPT.format(**violation_feedback)
        
        # 计算违反约束的数量，用于调整优化强度
        violation_count = sum(1 for v in constraint_violations.values() 
                            if v['verify_result'] == '不通过')
        
        # 获取自适应eta值
        adaptive_eta = self._get_adaptive_eta(iteration, violation_count)
        
        # 根据自适应eta动态调整优化策略
        if adaptive_eta < 0.3:  # 保守优化
            refinement_instruction = "进行最小化的精确修改，只针对验证失败的具体问题进行微调"
            base_temperature = 0.1
        elif adaptive_eta < 0.7:  # 中等优化  
            refinement_instruction = "进行适度的修改，在保持陷阱效果的前提下较大幅度地调整问题"
            base_temperature = 0.3
        else:  # 激进优化
            refinement_instruction = "进行较大幅度的重构，在保持核心陷阱的前提下显著改善问题质量"
            base_temperature = 0.5
        
        # 根据违反约束数量进一步调整温度
        adjusted_temperature = min(0.8, base_temperature + violation_count * 0.1)
        
        prompt = REFINE_PROMPT.format(
            raw_question=sample['question'],
            refer_diagnosis=sample['refer_diagnosis'],
            org_symptoms_lst=sample['org_symptoms_lst'],
            distractor_diagnosis=sample['distractor_diagnosis'],
            selected_symptoms=sample['selected_symptoms'],
            patient_desc=sample['patient_desc'],
            patient_style=sample['patient_style'],
            trap_question=sample["intermediate_results"]['trap_question'],
            misleading_knowledge=sample["intermediate_results"]['misleading_knowledge'],
            refinement_instruction=refinement_instruction,
            eta_value=adaptive_eta,
            reason=reason_text,
        )

        refinement_result = await self.call_api_json(
            model=self.generator_model,
            system_prompt=REFINE_SYSTEM_PROMPT,
            prompt=prompt,
            temperature=adjusted_temperature,  # 使用动态调整的温度
            json_schema=self.RefinementResult
        )

        # 应用优化步骤
        refined_sample = deepcopy(sample)
        refined_sample['question'] = refinement_result['refined_question']
        
        return refined_sample, refinement_result

    async def projected_gradient_descent(self, sample: Dict[str, Any], 
                                        overwrite: bool = False) -> Dict[str, Any]:
        """
        投影梯度下降主算法
        
        实现公式：q_{t+1} = Π_C(q_t - η∇L(q_t))
        
        Args:
            sample: 初始候选问题 q_0
            overwrite: 是否覆盖已有结果
            
        Returns:
            优化后的问题 q*（不动点）
        """
        question_id = sample['question_id']
        
        # 检查是否已有有效结果且不需要覆盖
        if not overwrite:
            existing_logs = self.logs_table.get(self.QueryModel.question_id == question_id)
            if existing_logs and existing_logs['logs']:
                last_log = existing_logs['logs'][-1]
                if (last_log['action_type'] == 'validation' and 
                    last_log['result']['is_in_feasible_set']):
                    return sample
        
        current_sample = deepcopy(sample)
        
        # 投影梯度下降迭代
        for t in range(self.max_iterations):
            # 步骤1: 验证当前候选是否在可行集C中
            is_in_feasible_set, constraint_violations = await self.validator(current_sample)
            
            self._log_iteration(
                question_id, t, "validation", 
                current_sample['question'],
                {"is_in_feasible_set": is_in_feasible_set, "violations": constraint_violations}
            )
            
            # 步骤2: 检查终止条件（到达不动点）
            if is_in_feasible_set:
                # 找到不动点 q* = Π_C(q*)，更新数据库
                self.questions_table.update(
                    {
                        'question': current_sample['question']
                    },
                    self.QueryModel.question_id == question_id
                )
                return current_sample
            
            # 步骤3: 应用优化步骤 q_t - η∇L(q_t)
            if t < self.max_iterations - 1:  # 避免最后一次迭代时不必要的优化
                refined_sample, refinement_info = await self.refinement_operator(
                    current_sample, constraint_violations, iteration=t
                )
                
                self._log_iteration(
                    question_id, t, "refinement",
                    refined_sample['question'],
                    refinement_info
                )
                
                # 步骤4: 投影到可行集（隐式投影，通过下一次验证实现）
                current_sample = refined_sample
        
        # 达到最大迭代次数仍未收敛
        return current_sample

    async def optimize_sample(self, sample: Dict[str, Any], overwrite: bool = False) -> None:
        """
        优化单个样本的包装函数，包含错误处理和进度显示
        """
        try:
            # with tqdm_asyncio(
            #     total=0, 
            #     desc=f"[PGD] Optimizing sample {sample['question_id']}", 
            #     ncols=75, 
            #     leave=False
            # ) as pbar:
            optimized_sample = await self.projected_gradient_descent(sample, overwrite)
                # pbar.set_description(
                #     f"[PGD] Sample {sample['question_id']} optimization completed"
                # )
            return optimized_sample
        except Exception as e:
            print(f"Error optimizing sample {sample['question_id']}: {e}")
            traceback.print_exc()
            return sample

    async def run(self, overwrite: bool = False, max_concurrency: int = 8):
        """
        主入口方法：批量运行投影梯度下降优化
        """
        self._init_caller()
        
        async with AIOTinyDB(
            self.data_path, 
            indent=4, 
            separators=(',', ': '), 
            ensure_ascii=False
        ) as db:
            self.questions_table = db.table('questions')
            self.logs_table = db.table('logs')

            if overwrite:
                self.logs_table.truncate()

            # 为每个样本创建优化任务
            optimization_tasks = [
                self.optimize_sample(sample, overwrite) 
                for sample in self.questions_table
            ]
            
            # 并发执行优化任务
            await run_with_semaphore(
                optimization_tasks, 
                max_concurrency,
                desc='[PGD] Running projected gradient descent optimization'
            )

