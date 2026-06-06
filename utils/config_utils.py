import importlib
import importlib.util
import os
from types import ModuleType


def _build_module_candidates(config_name: str):
    normalized = config_name.strip()
    if normalized.endswith(".py"):
        normalized = normalized[:-3]

    # Allow users to pass module-like paths (e.g., faf_cd.levir_dinov3_convnext_large)
    if "/" in normalized:
        normalized = normalized.replace("/", ".")

    if normalized.startswith("configs."):
        return [normalized]

    # Allow subpackage-qualified names (e.g., faf_cd.levir_dinov3_convnext_large)
    if "." in normalized:
        pkg, module = normalized.rsplit(".", 1)
        if module.startswith("config_"):
            module_candidates = [module]
        else:
            module_candidates = [f"config_{module}", module]
        return [f"configs.{pkg}.{name}" for name in module_candidates]

    if normalized.startswith("config_"):
        prefixed = normalized
        unprefixed = normalized[len("config_") :]
    else:
        prefixed = f"config_{normalized}"
        unprefixed = normalized

    candidates = []
    config_names = [prefixed, f"config_{unprefixed}"]
    seen = set()

    for name in config_names:
        root_candidate = f"configs.{name}"
        if root_candidate not in seen:
            candidates.append(root_candidate)
            seen.add(root_candidate)

        for subpkg in _discover_config_subpackages():
            subpkg_candidate = f"configs.{subpkg}.{name}"
            if subpkg_candidate not in seen:
                candidates.append(subpkg_candidate)
                seen.add(subpkg_candidate)

    return candidates


def _discover_config_subpackages():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs_dir = os.path.join(project_root, "configs")
    if not os.path.isdir(configs_dir):
        return []

    subpackages = []
    for name in sorted(os.listdir(configs_dir)):
        path = os.path.join(configs_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("_") or name == "__pycache__":
            continue
        subpackages.append(name)
    return subpackages


def _is_explicit_module_name(config_name: str) -> bool:
    normalized = config_name.strip()
    return normalized.startswith("configs.") or "/" in normalized or "." in normalized


def _existing_module_candidates(module_candidates):
    existing = []
    for module_name in module_candidates:
        if importlib.util.find_spec(module_name) is not None:
            existing.append(module_name)
    return existing


def _apply_config_defaults(cfg):
    if not hasattr(cfg, 'grad_accum_steps'):
        cfg.grad_accum_steps = 1
    if not hasattr(cfg, 'log_backend'):
        cfg.log_backend = 'tensorboard'
    if getattr(cfg, 'backbone', None) != 'dinov3':
        raise ValueError("FAF-CD release configs must use backbone='dinov3'.")
    if getattr(cfg, 'decoder', None) != 'MambaDecoder':
        raise ValueError("FAF-CD release configs must use decoder='MambaDecoder'.")
    return cfg


def _config_from_module(module: ModuleType, source: str):
    if not hasattr(module, "config"):
        raise ValueError(
            f"Config source '{source}' does not define a 'config' object."
        )
    return _apply_config_defaults(module.config)


def load_config_by_name(config_name: str):
    if not config_name:
        raise ValueError("Config name is required.")

    module_candidates = _build_module_candidates(config_name)

    if not _is_explicit_module_name(config_name):
        existing_candidates = _existing_module_candidates(module_candidates)
        if len(existing_candidates) > 1:
            options = ", ".join(existing_candidates)
            raise ValueError(
                f"Config name '{config_name}' is ambiguous. Matches: {options}. "
                "Please pass an explicit module name, e.g. 'faf_cd.<config_name>' "
                "or 'configs.faf_cd.<config_name>'."
            )

    last_error = None
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            return _config_from_module(module, module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                last_error = exc
                continue
            raise

    searched = ", ".join(f"{name}.py" for name in module_candidates)
    raise ValueError(
        f"Config '{config_name}' not found. Tried: {searched}"
    ) from last_error


def load_config_by_path(config_path: str):
    if not config_path:
        raise ValueError("Config path is required.")
    abs_path = os.path.abspath(config_path)
    if not os.path.isfile(abs_path):
        raise ValueError(f"Config file does not exist: {abs_path}")
    module_name = f"user_config_{abs(hash(abs_path))}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import config from path: {abs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _config_from_module(module, abs_path)
