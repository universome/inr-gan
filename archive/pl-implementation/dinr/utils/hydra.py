from typing import Any, Union, Callable

import hydra
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf


def call(config: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Implementation of recursive instantiation
    unless there is a better official way https://github.com/facebookresearch/hydra/issues/566#issuecomment-677713399

    :param config: An object describing what to call and what params to use. needs to have a _target_ field.
    :param args: optional positional parameters pass-through
    :param kwargs: optional named parameters pass-through
    :return: the return value from the specified class or method
    """
    if config is None:
        return None

    if isinstance(config, ListConfig) and isinstance(config[0], DictConfig) and '_target_' in config[0]:
        obj_list = []
        for conf_item in config:
            obj_list.append(call(conf_item))
        return obj_list
        # kwargs[key] = obj_list
        # primitive_conf = OmegaConf.to_container(config, resolve=True)
        # del primitive_conf[key]
        # config = DictConfig(primitive_conf)
        # return call(config, *args, **kwargs)

    for key, child_conf in config.items():
        if isinstance(child_conf, ListConfig) and isinstance(child_conf[0], DictConfig) and '_target_' in child_conf[0]:
            obj_list = []
            for conf_item in child_conf:
                obj_list.append(call(conf_item))
            kwargs[key] = obj_list
            primitive_conf = OmegaConf.to_container(config, resolve=True)
            del primitive_conf[key]
            config = DictConfig(primitive_conf)
            return call(config, *args, **kwargs)
        elif isinstance(child_conf, ListConfig) and not isinstance(child_conf[0], DictConfig) and not isinstance(
                child_conf[0], ListConfig):
            kwargs[key] = list(child_conf)
            primitive_conf = OmegaConf.to_container(config, resolve=True)
            del primitive_conf[key]
            config = DictConfig(primitive_conf)
            return call(config, *args, **kwargs)
        elif isinstance(child_conf, DictConfig) and '_target_' in child_conf and key not in kwargs:
            kwargs[key] = call(child_conf)
            primitive_conf = OmegaConf.to_container(config, resolve=True)
            del primitive_conf[key]
            config = DictConfig(primitive_conf)
            return call(config, *args, **kwargs)
        elif key != '_target_' and isinstance(child_conf, str):
            try:
                object = _locate(child_conf)
                kwargs[key] = object
                primitive_conf = OmegaConf.to_container(config, resolve=True)
                del primitive_conf[key]
                config = DictConfig(primitive_conf)
                return call(config, *args, **kwargs)
            except ImportError:
                # raise ValueError
                pass

    return hydra.utils.call(config, *args, **kwargs)


# Alias for call
instantiate = call


def _locate(path: str) -> Union[type, Callable[..., Any]]:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    import builtins
    from importlib import import_module

    parts = [part for part in path.split(".") if part]
    module = None
    for n in reversed(range(len(parts))):
        try:
            mod = ".".join(parts[:n])
            module = import_module(mod)
        except Exception as e:
            if n == 0:
                raise ImportError(f"Error loading module '{path}'") from e
            continue
        if module:
            break
    if module:
        obj = module
    else:
        obj = builtins
    for part in parts[n:]:
        mod = mod + "." + part
        if not hasattr(obj, part):
            try:
                import_module(mod)
            except Exception as e:
                raise ImportError(
                    f"Encountered error: `{e}` when loading module '{path}'"
                ) from e
        obj = getattr(obj, part)
    if isinstance(obj, type):
        obj_type: type = obj
        return obj_type
    elif callable(obj):
        obj_callable: Callable[..., Any] = obj
        return obj_callable
    else:
        # dummy case
        raise ValueError(f"Invalid type ({type(obj)}) found for {path}")
