"""
Utils package for DyReMe

This package contains utility modules for API calls, configuration loading, and model handling.
"""

"""
Utils package for DyReMe

This package contains utility modules for API calls, configuration loading, and model handling.
"""

# 新的推荐导入方式
from .model_caller import ModelCaller, run_with_semaphore
from .model_providers import ProviderFactory
from .load_config import load_keys_to_env, get_api_key

# # 向后兼容的导入
# from .common_utils import (
#     run_with_semaphore, SearchEngine, LOG_RECORDS, TOKENS_COUNT_DB_PATH,
#     # 保留原有的Caller以保证向后兼容
#     Caller
# )

# from .common_utils import run_with_semaphore

load_keys_to_env()

__all__ = [
    'ModelCaller', 'default_caller', 'ProviderFactory',
    'load_keys_to_env', 'get_api_key',
    'run_with_semaphore', 'SearchEngine', 'LOG_RECORDS', 'TOKENS_COUNT_DB_PATH',
    'Caller'  # 向后兼容
]