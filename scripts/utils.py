from pathlib import Path
from typing import Dict

import yaml


def write_yaml_file(data: Dict, filepath: Path):
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
