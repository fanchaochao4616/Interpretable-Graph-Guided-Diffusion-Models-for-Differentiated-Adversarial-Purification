import yaml
from types import SimpleNamespace
def load_config(yaml_path):
    """加载 YAML 文件并转换为 SimpleNamespace 对象"""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    return dict_to_namespace(config_dict)