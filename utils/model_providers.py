"""
模型提供者抽象层
将不同的API提供者封装成统一的接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from openai import AsyncOpenAI, AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential 
import json
import json_repair
from pydantic import ValidationError

from .load_config import get_api_key
import os


class BaseModelProvider(ABC):
    """模型提供者基类"""
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self._client = None
    
    @property
    def client(self):
        """懒加载客户端"""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    @abstractmethod
    def _create_client(self):
        """创建API客户端"""
        pass
    
    @abstractmethod
    async def call_api(self, model: str, system_prompt: str, prompt: str, 
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        """调用API获取文本响应"""
        pass
    
    async def call_api_json(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0, 
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """调用API获取JSON响应"""
        raise NotImplementedError("JSON mode not supported for this provider")
    
    async def call_api_web(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> Tuple[str, Any]:
        """调用Web API获取文本响应和引用"""
        raise NotImplementedError("Web mode not supported for this provider")

    async def call_api_json_web(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0,
                           max_tokens: Optional[int] = None) -> Tuple[Dict[str, Any], Any]:
        """调用Web API获取JSON响应和引用"""
        raise NotImplementedError("Web JSON mode not supported for this provider")


class DeepSeekProvider(BaseModelProvider):
    """DeepSeek模型提供者"""
    
    def _create_client(self):
        return AsyncOpenAI(
            api_key=get_api_key("DEEPSEEK_API_KEY"),
            base_url='XXX'
        )
    
    async def call_api(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return completion.choices[0].message.content
    
    async def call_api_json(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0,
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        max_tokens = max_tokens or self.max_tokens
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            response_format={'type': 'json_object'}
        )
        return json.loads(completion.choices[0].message.content)


class QwenProvider(BaseModelProvider):
    """Qwen模型提供者"""
    
    def _create_client(self):
        return AsyncOpenAI(
            api_key=get_api_key("QWEN_API_KEY"),
            base_url="XXX"
        )
    
    async def call_api(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return completion.choices[0].message.content


class DoubaoProvider(BaseModelProvider):
    """豆包模型提供者"""
    
    def _create_client(self):
        return AsyncOpenAI(
            api_key=get_api_key("DOUBAO_API_KEY"),
            base_url="XXX"
        )
    
    async def call_api(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return completion.choices[0].message.content
    
    async def call_api_json(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0,
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        max_tokens = max_tokens or self.max_tokens
        JSON_PREFILL_PREFIX = '{'
        
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": JSON_PREFILL_PREFIX}
            ],
            temperature=temperature
        )
        
        json_string = JSON_PREFILL_PREFIX + completion.choices[0].message.content
        try:
            obj = json.loads(json_string)
        except json.JSONDecodeError:
            obj = json_repair.loads(json_string)
        
        if json_schema:
            try:
                json_schema(**obj)  # 验证格式
            except ValidationError as e:
                print(f"Validation error: {e}")
                print(f"Invalid JSON: {obj}")
                raise e
        
        return obj


class DoubaoWebProvider(BaseModelProvider):
    """豆包Web模型提供者"""
    
    def _create_client(self):
        return AsyncOpenAI(
            api_key=get_api_key("DOUBAO_API_KEY"),
            base_url="XXX"
        )
    
    async def call_api(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return completion.choices[0].message.content
    
    async def call_api_web(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> Tuple[str, Any]:
        max_tokens = max_tokens or self.max_tokens
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        references = completion.references if hasattr(completion, 'references') else None
        return completion.choices[0].message.content, references

    async def call_api_json_web(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0,
                           max_tokens: Optional[int] = None) -> Tuple[Dict[str, Any], Any]:
        max_tokens = max_tokens or self.max_tokens
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            response_format={'type': 'json_object'}
        )
        # import pdb; pdb.set_trace()
        references = completion.references if hasattr(completion, 'references') else None
    
        obj = json_repair.loads(completion.choices[0].message.content)
        
        if json_schema:
            try:
                json_schema(**obj)  # 验证格式
            except ValidationError as e:
                print(f"Validation error: {e}")
                print(f"Invalid JSON: {obj}")
                raise e
        
        return obj, references

class ExternalAzureOpenAIProvider(BaseModelProvider):
    """Azure OpenAI模型提供者"""

    def __init__(self, max_tokens: int = 4096):
        super().__init__(max_tokens)
    
    def _create_client(self):

        return AsyncOpenAI(
            api_key=os.getenv("GPT_KEY"),
            base_url=os.getenv('GPT_ENDPOINT')

        )
    
    async def call_api(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return completion.choices[0].message.content
    
    async def call_api_json(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0,
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        max_tokens = max_tokens or self.max_tokens
        # import pdb; pdb.set_trace()
        try:
            completion = await self.client.beta.chat.completions.parse(
                model=model,
                max_tokens=max_tokens,
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format=json_schema
            )
            # import pdb; pdb.set_trace()

            # obj = json_repair.loads(completion.choices[0].message.content)
            return completion.choices[0].message.parsed.model_dump()
        except Exception as e:
            # import pdb; pdb.set_trace()
            completion = await self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={'type': 'json_object'}
            )
            obj = json_repair.loads(completion.choices[0].message.content)
            return obj


        # json_repair

        # return json.loads(completion.choices[0].message.content)

class AzureOpenAIProvider(BaseModelProvider):
    """Azure OpenAI模型提供者"""

    def __init__(self, endpoint: str, max_tokens: int = 4096):
        super().__init__(max_tokens)
        self.endpoint = endpoint
    
    def _create_client(self):
        token_provider = get_bearer_token_provider(
            AzureCliCredential(),
            "XXX"
        )
        return AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview"
        )
    
    async def call_api(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        if model in ['o1-mini', 'o1']:
            completion = await self.client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens * 4,   # for cot
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": system_prompt + '\n\n' + prompt}
                ],
                # temperature=max(temperature, 0.6)  # o1-mini最低为0.6
            )
        else:
            completion = await self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
        return completion.choices[0].message.content
    
    async def call_api_json(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0,
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        max_tokens = max_tokens or self.max_tokens
        if model in ['o1-mini', 'o1']:
            completion = await self.client.beta.chat.completions.parse(
                model=model,
                max_completion_tokens=max_tokens * 4,   # for cot
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": system_prompt + '\n\n' + prompt}
                ],
                # temperature=max(temperature, 0.6),  # o1-mini最低为0.6
                response_format=json_schema
            )
            return completion.choices[0].message.parsed.model_dump()
        else:
            completion = await self.client.beta.chat.completions.parse(
                model=model,
                max_tokens=max_tokens,
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format=json_schema
            )
            obj = json_repair.loads(completion.choices[0].message.content)
            return obj
        
        # json_repair

        # return json.loads(completion.choices[0].message.content)


class VLLMProvider(BaseModelProvider):
    """VLLM本地模型提供者"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", max_tokens: int = 4096):
        super().__init__(max_tokens)
        self.base_url = base_url
    
    def _create_client(self):
        return AsyncOpenAI(
            api_key="EMPTY",
            base_url=self.base_url
        )
    
    async def call_api(self, model: str, system_prompt: str, prompt: str,
                      temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        
        # 某些模型可能不支持system prompt
        if 'FreedomIntelligence' in model:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        
        completion = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature
        )
        return completion.choices[0].message.content
    
    async def call_api_json(self, model: str, system_prompt: str, prompt: str,
                           json_schema=None, temperature: float = 0.0,
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        max_tokens = max_tokens or self.max_tokens
        
        # 某些模型可能不支持system prompt
        if 'HuatuoGPT2-7B' in model:
            messages = [{"role": "user", "content": system_prompt + '\n\n' + prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        
        try:
            # 尝试使用 response_format 参数（如果VLLM支持的话）
            # completion = await self.client.chat.completions.create(
            #     model=model,
            #     max_tokens=max_tokens,
            #     messages=messages,
            #     temperature=temperature,
            #     response_format=json_schema
            # )
            # obj = json.loads(completion.choices[0].message.content)

            completion = await self.client.beta.chat.completions.parse(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature,
                response_format=json_schema
            )
            obj = completion.choices[0].message.parsed.model_dump()
        except Exception:
            # 如果不支持 response_format，回退到普通请求然后解析JSON
            completion = await self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature
            )
            try:
                obj = json.loads(completion.choices[0].message.content)
            except json.JSONDecodeError:
                obj = json_repair.loads(completion.choices[0].message.content)
        
        if json_schema:
            try:
                json_schema(**obj)  # 验证格式
            except ValidationError as e:
                print(f"Validation error: {e}")
                print(f"Invalid JSON: {obj}")
                raise e
        
        return obj


# 提供者工厂
class ProviderFactory:
    """模型提供者工厂"""
    
    _providers = {
        # 'deepseek': DeepSeekProvider,
        'external_azure': ExternalAzureOpenAIProvider,
        'qwen': QwenProvider,
        'doubao': DoubaoProvider,
        'doubao_web': DoubaoWebProvider,
        # 'vllm': VLLMProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, **kwargs) -> BaseModelProvider:
        """创建模型提供者实例"""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        if callable(provider_class) and not isinstance(provider_class, type):
            # 处理lambda函数
            return provider_class()
        else:
            return provider_class(**kwargs)
    
    @classmethod
    def get_provider_for_model(cls, model_name: str, **kwargs) -> BaseModelProvider:
        """根据模型名称自动选择提供者"""
        model_lower = model_name.lower()
        
        # 如果提供了base_url参数，说明是本地部署的模型，使用VLLM提供者
        if 'base_url' in kwargs:
            return cls.create_provider('vllm', **kwargs)
        elif 'DeepSeek-V3-0324' in model_name:
            return cls.create_provider('external', **kwargs)
        elif 'qwen' in model_lower:
            return cls.create_provider('qwen', **kwargs)
        elif 'ep-' in model_name:  # 豆包endpoint ID
            return cls.create_provider('doubao', **kwargs)
        elif 'bot-' in model_name:  # 豆包web bot
            return cls.create_provider('doubao_web', **kwargs)

        else:
            # 默认使用VLLM（本地模型）
            return cls.create_provider('vllm', **kwargs)
