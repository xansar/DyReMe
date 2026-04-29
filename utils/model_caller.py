"""
简化的模型调用器
负责并发控制、重试逻辑和错误处理
"""
import asyncio
import time
from typing import Dict, Any, Optional
from .model_providers import BaseModelProvider, ProviderFactory
import traceback


class ModelCaller:
    """统一的模型调用器"""
    
    def __init__(self, max_concurrent_requests: int = 8, max_retries: int = 10, retry_delay: int = 15):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._providers_cache = {}  # 缓存提供者实例
    
    def _get_provider(self, model_name: str, provider_kwargs: Optional[Dict] = None) -> BaseModelProvider:
        """获取或创建模型提供者"""
        provider_kwargs = provider_kwargs or {}
        cache_key = (model_name, tuple(sorted(provider_kwargs.items())))
        
        if cache_key not in self._providers_cache:
            self._providers_cache[cache_key] = ProviderFactory.get_provider_for_model(
                model_name, **provider_kwargs
            )
        return self._providers_cache[cache_key]
    
    async def call_api(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None,
                      provider_kwargs: Optional[Dict] = None) -> Optional[str]:
        """
        调用模型API获取文本响应
        
        Args:
            model: 模型名称
            system_prompt: 系统提示
            prompt: 用户提示
            temperature: 温度参数
            max_tokens: 最大token数
            provider_kwargs: 提供者特定参数（如VLLM的base_url）
        
        Returns:
            模型响应文本，失败时返回None
        """
        async with self.semaphore:
            provider = self._get_provider(model, provider_kwargs)
            
            for attempt in range(self.max_retries):
                try:
                    return await provider.call_api(
                        model=model,
                        system_prompt=system_prompt,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    traceback.print_exc()
                    attempt += 1
                    print(f"Attempt {attempt} failed for model {model}: {e}")
                    
                    if attempt < self.max_retries:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        print(f"Max retries reached for model {model}. Aborting.")
                        return None
    
    async def call_api_json(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0,
                           max_tokens: Optional[int] = None,
                           provider_kwargs: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        调用模型API获取JSON响应
        """
        async with self.semaphore:
            provider = self._get_provider(model, provider_kwargs)
            
            for attempt in range(self.max_retries):
                # import pdb; pdb.set_trace()
                try:
                    return await provider.call_api_json(
                        model=model,
                        system_prompt=system_prompt,
                        prompt=prompt,
                        json_schema=json_schema,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    traceback.print_exc()
                    attempt += 1
                    print(f"Attempt {attempt} failed for model {model}: {e}")
                    print('=' * 50)
                    print(prompt)
                    print('=' * 50)
                    
                    if attempt < self.max_retries:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        print(f"Max retries reached for model {model}. Aborting.")
                        raise e
    
    async def call_api_web(self, model: str, system_prompt: str, prompt: str,
                          temperature: float = 0.0, max_tokens: Optional[int] = None,
                          provider_kwargs: Optional[Dict] = None):
        """
        调用Web模型API（返回内容和引用）
        """
        async with self.semaphore:
            provider = self._get_provider(model, provider_kwargs)
            
            # 确保是Web提供者
            if not hasattr(provider, '__class__') or 'Web' not in provider.__class__.__name__:
                raise Exception(f"Model {model} does not support web mode")
            
            for attempt in range(self.max_retries):
                try:
                    return await provider.call_api_web(
                        model=model,
                        system_prompt=system_prompt,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    traceback.print_exc()
                    if 'does not support web mode' in str(e):
                        raise e
                    
                    attempt += 1
                    print(f"Attempt {attempt} failed for model {model}: {e}")
                    
                    if attempt < self.max_retries:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        print(f"Max retries reached for model {model}. Aborting.")
                        raise e

    async def call_api_json_web(self, model: str, system_prompt: str, prompt: str,
                               json_schema=None, temperature: float = 0.0,
                               max_tokens: Optional[int] = None,
                               provider_kwargs: Optional[Dict] = None):
        """
        调用Web模型API获取JSON响应
        """
        async with self.semaphore:
            provider = self._get_provider(model, provider_kwargs)
            
            # 确保是Web提供者
            if not hasattr(provider, '__class__') or 'Web' not in provider.__class__.__name__:
                raise Exception(f"Model {model} does not support web mode")
            
            for attempt in range(self.max_retries):
                try:
                    return await provider.call_api_json_web(
                        model=model,
                        system_prompt=system_prompt,
                        prompt=prompt,
                        json_schema=json_schema,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    if 'does not support web mode' in str(e):
                        raise e
                    traceback.print_exc()
                    attempt += 1
                    print(f"Attempt {attempt} failed for model {model}: {e}")
                    
                    
                    if attempt < self.max_retries:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        print(f"Max retries reached for model {model}. Aborting.")
                        raise e


# 创建默认实例
# default_caller = ModelCaller()

# # 为了向后兼容，保留原有的Caller类
# Caller = ModelCaller


async def run_with_semaphore(tasks, max_concurrency, desc=None, position=0):
    """
    使用信号量控制并发执行异步任务
    
    Args:
        tasks: 异步任务列表
        max_concurrency: 最大并发数
        desc: 进度条描述
        position: 进度条位置
    
    Returns:
        任务执行结果列表
    """
    import asyncio
    from tqdm.asyncio import tqdm_asyncio
    
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _run_task_with_semaphore(task):
        async with semaphore:
            return await task
    
    # 包装所有任务
    wrapped_tasks = [_run_task_with_semaphore(task) for task in tasks]
    
    # 使用 tqdm 显示进度
    if desc:
        return await tqdm_asyncio.gather(*wrapped_tasks, desc=desc, position=position)
    else:
        return await tqdm_asyncio.gather(*wrapped_tasks)