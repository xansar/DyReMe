#!/usr/bin/env python3
"""
配置加载工具：将 key.yaml 文件中的密钥加载为环境变量
"""

import os
import yaml
from pathlib import Path

def load_keys_to_env(config_file='./utils/key.yaml'):
    """
    从 YAML 配置文件中加载 API 密钥到环境变量
    
    Args:
        config_file (str): 配置文件路径，默认为 'key.yaml'
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        print(f"警告: 配置文件 {config_file} 不存在")
        return
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config:
            for key, value in config.items():
                # 只设置以 _API_KEY 结尾的配置项
                if key.endswith('_API_KEY'):
                    os.environ[key] = str(value)
                    print(f"已加载环境变量: {key}")
        
        print(f"配置加载完成，共加载 {len([k for k in config.keys() if k.endswith('_API_KEY')])} 个 API 密钥")
                    
    except yaml.YAMLError as e:
        print(f"错误: 解析 YAML 文件失败: {e}")
    except Exception as e:
        print(f"错误: 加载配置文件失败: {e}")

def get_api_key(key_name):
    """
    获取 API 密钥，优先从环境变量获取，如果没有则从配置文件加载
    
    Args:
        key_name (str): 密钥名称，如 'DEEPSEEK_API_KEY'
    
    Returns:
        str: API 密钥值
    """
    # 先尝试从环境变量获取
    key_value = os.environ.get(key_name)
    
    if key_value and key_value != key_name:  # 确保不是占位符
        return key_value
    
    # 如果环境变量中没有，尝试从配置文件加载
    try:
        config_path = Path('./key.yaml')
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config and key_name in config:
                    return config[key_name]
    except Exception as e:
        print(f"警告: 无法从配置文件读取 {key_name}: {e}")
    
    # 返回原始值作为后备
    return key_name

if __name__ == '__main__':
    # 测试加载配置
    load_keys_to_env()
