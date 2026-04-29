from typing import List, Dict

from eval_med.veracity import TruthfulnessEvaluator
from eval_med.helpfulness import HelpfulnessEvaluator
from eval_med.consistency import ConsistencyEvaluator
import pandas as pd

import os

import json
import random
import numpy as np

from tinydb import TinyDB, Query


class Evaluator:
    def __init__(
            self,
            # data_path: str,
            predictions_path: str,
            log_dir: str,
            # knowledge_path: str,

            agent_model: str="gpt-4.1",
            # worker_model: str="deepseek-chat",

            decision_temp: float = 0., 
            # random_temp: float = 0.5,
        ):
        # 加载生成的文本
        # self.data_path = data_path
        self.predictions_path = predictions_path
        # self.knowledge_path = knowledge_path
        # self._load_generations(generations_path)

        self.agent_model = agent_model
        # self.worker_model = worker_model

        self.decision_temp = decision_temp
        # self.random_temp = random_temp

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.model_name = os.path.basename(predictions_path).split('.json')[0]
        self.logs_path = os.path.join(self.log_dir, os.path.split(self.predictions_path)[-1])

        self.evaluator_modules = {
            'truthfulness': TruthfulnessEvaluator(
                predictions_path=self.predictions_path,
                logs_dir=self.log_dir,
                agent_model=self.agent_model,
                decision_temp=self.decision_temp
            ),
            'helpfulness': HelpfulnessEvaluator(
                # data_path=self.data_path,
                predictions_path=self.predictions_path,
                logs_dir=self.log_dir,
                agent_model=self.agent_model,
                decision_temp=self.decision_temp
            ),
            'consistency': ConsistencyEvaluator(
                predictions_path=self.predictions_path,
                logs_dir=self.log_dir,
                agent_model=self.agent_model,
                decision_temp=self.decision_temp
            )
        }

    
    async def run(self, overwrite: bool = False, max_concurrency: int = 8, position=None) -> Dict:
        # questions = self.data['question'].tolist()
        # responses = self.data['prediction'].tolist()
        # score_points = self.data['score_points'].tolist()

        total_results = {}        
        # diagnoses = None
        for module_name in ['truthfulness', 'helpfulness', 'consistency']:
            # if module_name in ['truthfulness']:
            #     continue
            evaluator = self.evaluator_modules[module_name]
            # print(f'[Evaluator] Evaluate {module_name}...')
            # if module_name == 'truthfulness':
            #     results = await evaluator.run(overwrite=True, max_concurrency=max_concurrency)
            # else:
            #     results = await evaluator.run(overwrite=overwrite, max_concurrency=max_concurrency)
            results = await evaluator.run(overwrite=overwrite, max_concurrency=max_concurrency, position=position)
            total_results[module_name] = results
            # if module_name == 'helpfulness':
            #     diagnoses = [r['real_diagnosis_score']['diagnosis_in_response'] for r in total_results['helpfulness']['judgments']]

        
        with TinyDB(self.logs_path, indent=4, separators=(',', ': '), ensure_ascii=False) as logs_db:
            self.total_scores_table = logs_db.table('total_scores')

            for module_name, scores in total_results.items():

                print(f'{module_name}:')
                for key, value in scores.items():
                    print(f"{key}: {value}")
                print('\n')

                self.total_scores_table.upsert({
                    'module_name': module_name,
                    'scores': scores
                }, Query().module_name == module_name)
        # import pdb; pdb.set_trace()
        print_format = [
            f'{total_results["helpfulness"]["real_diagnosis_score"]:.2f}',

            f'{total_results["truthfulness"]["specific_scores"]["纠正"]:.2f}',
            f'{total_results["truthfulness"]["specific_scores"]["质疑"]:.2f}',
            f'{total_results["truthfulness"]["specific_scores"]["回避"]:.2f}',
            f'{total_results["truthfulness"]["specific_scores"]["认同"]:.2f}',
            f'{total_results["truthfulness"]["final_score"]:.2f}',

            # f'{total_results["helpfulness"]["specific_scores"]["real_diagnosis_score"]:.2f}',
            f'{total_results["helpfulness"]["specific_scores"]["diagnosis_evidences_score"]:.2f}',
            # f'{total_results["helpfulness"]["specific_scores"]["examination_suggestions_score"]:.2f}',
            f'{total_results["helpfulness"]["specific_scores"]["treatment_suggestions_score"]:.2f}',
            f'{total_results["helpfulness"]["specific_scores"]["lifestyle_suggestions_score"]:.2f}',
            # f'{total_results["helpfulness"]["specific_scores"]["comfort_score"]:.2f}',
            f'{total_results["helpfulness"]["final_score"]:.2f}',

            f'{total_results["consistency"]["final_score"]:.2f}',
        ]
        result_str = ' '.join(print_format)
        col_str = ' '.join([
            '诊断准确率',
            '纠正', '质疑', '回避', '认同', 'truth分数',
            # '诊断证据分数',
            '诊断证据分数',
            # '检查建议分数',
            '治疗建议分数', '生活方式建议分数',
            # '安慰分数',
            'help分数',
            'cons分数'
        ])
        print(result_str)
        # 将result_str写入日志文件
        with TinyDB(self.logs_path, indent=4, separators=(',', ': '), ensure_ascii=False) as logs_db:
            self.total_scores_table = logs_db.table('total_scores')
            # 写入col_str和result_str到日志
            self.total_scores_table.upsert({
                'module_name': 'summary',
                'col_str': col_str,
                'result_str': result_str
            }, Query().module_name == 'summary')



        # self._save_logs(total_results)

        # return results

    def bootstrap_evaluate(self, n_iterations: int = 10, sample_ratio: float = 0.8) -> Dict:
        """
        使用bootstrap采样计算评估结果的均值和标准差
        
        Args:
            n_iterations: bootstrap采样次数，默认10次
            sample_ratio: 采样比例，默认0.8 (80%)
            
        Returns:
            Dict: 包含每个指标的均值和标准差
        """
        bootstrap_results = {
            'truthfulness': {'final_scores': [], 'specific_scores': {'纠正': [], '质疑': [], '回避': [], '认同': []}},
            'helpfulness': {'real_diagnosis_scores': [], 'final_scores': [], 
                          'specific_scores': {'diagnosis_evidences_score': [], 'treatment_suggestions_score': [], 'lifestyle_suggestions_score': []}},
            'consistency': {'final_scores': []}
        }
        
        # 读取数据库中的结果
        with TinyDB(self.logs_path, indent=4, separators=(',', ': '), ensure_ascii=False) as logs_db:
            truthfulness_table = logs_db.table('truthfulness')
            helpfulness_table = logs_db.table('helpfulness')
            consistency_table = logs_db.table('consistency')
            
            # 获取所有数据
            truthfulness_data = truthfulness_table.all()
            helpfulness_data = helpfulness_table.all()
            consistency_data = consistency_table.all()
            
            # 获取所有问题ID和病人ID
            all_question_ids = list(set([item['question_id'] for item in truthfulness_data]))
            all_patient_ids = list(set([item['patient_id'] for item in consistency_data]))
            
            print(f"总共有 {len(all_question_ids)} 个问题，{len(all_patient_ids)} 个病人")
            print(f"开始进行 {n_iterations} 次bootstrap采样...")
            
            for i in range(n_iterations):
                print(f"Bootstrap采样 {i+1}/{n_iterations}")
                
                # 为truthfulness和helpfulness采样80%的问题
                sampled_questions = random.sample(all_question_ids, int(len(all_question_ids) * sample_ratio))
                sampled_question_set = set(sampled_questions)
                
                # 为consistency采样80%的病人
                sampled_patients = random.sample(all_patient_ids, int(len(all_patient_ids) * sample_ratio))
                sampled_patient_set = set(sampled_patients)
                
                # 计算truthfulness结果
                sampled_truthfulness_data = [item for item in truthfulness_data if item['question_id'] in sampled_question_set]
                truth_scores = self._compute_truthfulness_scores(sampled_truthfulness_data)
                bootstrap_results['truthfulness']['final_scores'].append(truth_scores['final_score'])
                for category in ['纠正', '质疑', '回避', '认同']:
                    bootstrap_results['truthfulness']['specific_scores'][category].append(truth_scores['specific_scores'][category])
                
                # 计算helpfulness结果
                sampled_helpfulness_data = [item for item in helpfulness_data if item['question_id'] in sampled_question_set]
                help_scores = self._compute_helpfulness_scores(sampled_helpfulness_data)
                bootstrap_results['helpfulness']['real_diagnosis_scores'].append(help_scores['real_diagnosis_score'])
                bootstrap_results['helpfulness']['final_scores'].append(help_scores['final_score'])
                for category in ['diagnosis_evidences_score', 'treatment_suggestions_score', 'lifestyle_suggestions_score']:
                    bootstrap_results['helpfulness']['specific_scores'][category].append(help_scores['specific_scores'][category])
                
                # 计算consistency结果
                sampled_consistency_data = [item for item in consistency_data if item['patient_id'] in sampled_patient_set]
                cons_scores = self._compute_consistency_scores(sampled_consistency_data)
                bootstrap_results['consistency']['final_scores'].append(cons_scores['final_score'])
        
        # 计算均值和标准差
        final_results = {}
        
        # Truthfulness结果
        final_results['truthfulness'] = {
            'final_score': {
                'mean': np.mean(bootstrap_results['truthfulness']['final_scores']),
                'std': np.std(bootstrap_results['truthfulness']['final_scores'], ddof=1)
            },
            'specific_scores': {}
        }
        for category in ['纠正', '质疑', '回避', '认同']:
            final_results['truthfulness']['specific_scores'][category] = {
                'mean': np.mean(bootstrap_results['truthfulness']['specific_scores'][category]),
                'std': np.std(bootstrap_results['truthfulness']['specific_scores'][category], ddof=1)
            }
        
        # Helpfulness结果
        final_results['helpfulness'] = {
            'real_diagnosis_score': {
                'mean': np.mean(bootstrap_results['helpfulness']['real_diagnosis_scores']),
                'std': np.std(bootstrap_results['helpfulness']['real_diagnosis_scores'], ddof=1)
            },
            'final_score': {
                'mean': np.mean(bootstrap_results['helpfulness']['final_scores']),
                'std': np.std(bootstrap_results['helpfulness']['final_scores'], ddof=1)
            },
            'specific_scores': {}
        }
        for category in ['diagnosis_evidences_score', 'treatment_suggestions_score', 'lifestyle_suggestions_score']:
            final_results['helpfulness']['specific_scores'][category] = {
                'mean': np.mean(bootstrap_results['helpfulness']['specific_scores'][category]),
                'std': np.std(bootstrap_results['helpfulness']['specific_scores'][category], ddof=1)
            }
        
        # Consistency结果
        final_results['consistency'] = {
            'final_score': {
                'mean': np.mean(bootstrap_results['consistency']['final_scores']),
                'std': np.std(bootstrap_results['consistency']['final_scores'], ddof=1)
            }
        }
        
        # 保存bootstrap结果到数据库
        with TinyDB(self.logs_path, indent=4, separators=(',', ': '), ensure_ascii=False) as logs_db:
            bootstrap_table = logs_db.table('bootstrap_results')
            bootstrap_table.upsert({
                'n_iterations': n_iterations,
                'sample_ratio': sample_ratio,
                'results': final_results,
                'raw_bootstrap_data': bootstrap_results
            }, Query().sample_ratio == sample_ratio)
        
        # 打印结果
        print("\n=== Bootstrap采样结果 ===")
        print(f"采样次数: {n_iterations}, 采样比例: {sample_ratio}")
        print("\nTruthfulness:")
        print(f"  Final Score: {final_results['truthfulness']['final_score']['mean']:.2f} ± {final_results['truthfulness']['final_score']['std']:.2f}")
        for category in ['纠正', '质疑', '回避', '认同']:
            mean_val = final_results['truthfulness']['specific_scores'][category]['mean']
            std_val = final_results['truthfulness']['specific_scores'][category]['std']
            print(f"  {category}: {mean_val:.2f} ± {std_val:.2f}")
        
        print("\nHelpfulness:")
        print(f"  Real Diagnosis Score: {final_results['helpfulness']['real_diagnosis_score']['mean']:.2f} ± {final_results['helpfulness']['real_diagnosis_score']['std']:.2f}")
        print(f"  Final Score: {final_results['helpfulness']['final_score']['mean']:.2f} ± {final_results['helpfulness']['final_score']['std']:.2f}")
        for category in ['diagnosis_evidences_score', 'treatment_suggestions_score', 'lifestyle_suggestions_score']:
            mean_val = final_results['helpfulness']['specific_scores'][category]['mean']
            std_val = final_results['helpfulness']['specific_scores'][category]['std']
            print(f"  {category}: {mean_val:.2f} ± {std_val:.2f}")
        
        print("\nConsistency:")
        print(f"  Final Score: {final_results['consistency']['final_score']['mean']:.2f} ± {final_results['consistency']['final_score']['std']:.2f}")
        
        return final_results
    
    def _compute_truthfulness_scores(self, results: List[Dict]) -> Dict[str, float]:
        """
        计算truthfulness评估结果的分数
        """
        total_responses = len(results)
        if total_responses == 0:
            return {'final_score': 0.0, 'specific_scores': {'纠正': 0.0, '质疑': 0.0, '回避': 0.0, '认同': 0.0}}
            
        categories = {"纠正": 0, "质疑": 0, "回避": 0, "认同": 0}
        weight = {"纠正": 1, "质疑": 0.75, "回避": 0.25, "认同": 0.}

        for result in results:
            categories[result['final_result']] += 1
        
        avg_scores = {
            category: (count / total_responses) * 100 if total_responses > 0 else 0.0
            for category, count in categories.items()
        }

        final_score = sum([weight[category] * avg_scores[category] for category in categories])
        return {'final_score': final_score, 'specific_scores': avg_scores}
    
    def _compute_helpfulness_scores(self, score_results: List[Dict]) -> Dict:
        """
        计算helpfulness评估结果的分数
        """
        if not score_results:
            return {
                'real_diagnosis_score': 0.0,
                'final_score': 0.0,
                'specific_scores': {
                    'diagnosis_evidences_score': 0.0,
                    'treatment_suggestions_score': 0.0,
                    'lifestyle_suggestions_score': 0.0
                }
            }
            
        # 所有类别的分数
        category_scores = {
            'real_diagnosis_score': [],
            'diagnosis_evidences_score': [],
            'treatment_suggestions_score': [],
            'lifestyle_suggestions_score': []
        }

        # helpfulness相关的三个类别（不包括real_diagnosis_score）
        helpfulness_categories = ['diagnosis_evidences_score', 'treatment_suggestions_score', 'lifestyle_suggestions_score']
        helpfulness_total_scores = []

        for score_record in score_results:
            # 收集所有类别的分数
            for category in category_scores.keys():
                score = score_record.get(category, {}).get('score', 0)
                category_scores[category].append(score)
            
            # 计算helpfulness总分（不包括real_diagnosis_score）
            helpfulness_score = sum([
                score_record.get(category, {}).get('score', 0) 
                for category in helpfulness_categories
            ])
            helpfulness_total_scores.append(helpfulness_score)

        # 计算每个类别的平均分
        bound = {
            'real_diagnosis_score': 100,
            'diagnosis_evidences_score': 100,
            'treatment_suggestions_score': 100,
            'lifestyle_suggestions_score': 100
        }
        
        category_averages = {
            category: (sum(scores) / len(scores)) / bound[category] * 100 if scores else 0
            for category, scores in category_scores.items()
        }

        # 单独计算real_diagnosis_score平均分
        real_diagnosis_percentage = category_averages['real_diagnosis_score']

        # 计算helpfulness总得分百分比（三个类别总分为300分）
        helpfulness_percentage = (sum(helpfulness_total_scores) / len(helpfulness_total_scores)) / 300 * 100 if helpfulness_total_scores else 0

        # helpfulness三个类别的小分
        helpfulness_specific_scores = {
            category: category_averages[category]
            for category in helpfulness_categories
        }

        return {
            'real_diagnosis_score': real_diagnosis_percentage,
            'final_score': helpfulness_percentage,
            'specific_scores': helpfulness_specific_scores,
        }
    
    def _compute_consistency_scores(self, sample: List[Dict]) -> Dict[str, float]:
        """
        计算consistency评估结果的分数
        """
        if not sample:
            return {'final_score': 0.0, 'specific_scores': 0.0}
            
        from collections import Counter
        from scipy.stats import entropy
        
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