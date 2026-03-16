from importlib import import_module
_mod = import_module("utils.evaluation_utils")  # top-level module in your repo
globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith("_")})
