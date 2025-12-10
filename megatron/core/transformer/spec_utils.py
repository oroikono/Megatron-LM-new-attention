# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import types
from dataclasses import dataclass, field
from typing import Tuple, Union


@dataclass
class ModuleSpec:
    """This is a Module Specification dataclass.

    Specification defines the location of the module (to import dynamically)
    or the imported module itself. It also defines the params that need to be
    passed to initialize the module.

    Args:
        module (Union[Tuple, type]): A tuple describing the location of the
            module class e.g. `(module.location, ModuleClass)` or the imported
            module class itself e.g. `ModuleClass` (which is already imported
            using `from module.location import ModuleClass`).
        params (dict): A dictionary of params that need to be passed while init.

    """

    module: Union[Tuple, type]
    params: dict = field(default_factory=lambda: {})
    submodules: type = None


def import_module(module_path: Tuple[str]):
    """Import a named object from a module in the context of this function.

    `module_path` can be provided in multiple equivalent forms:
      * (base_path, name) tuple or list, e.g. ("pkg.mod", "fn")
      * ["base_path", "name"] list from argparse `--spec base_path name`
      * single dotted string "base_path.name" from `--spec base_path.name`
    """

    # Normalize `module_path` to a (base_path, name) pair.
    if isinstance(module_path, (list, tuple)):
        if len(module_path) == 2:
            base_path, name = module_path
        elif len(module_path) == 1:
            # Support the common case where argparse `nargs='*'` captured a
            # single dotted string such as
            #   --spec megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_semigroup_spec
            dotted = module_path[0]
            if not isinstance(dotted, str) or "." not in dotted:
                raise ValueError(
                    f"spec must be provided as '<module>.<name>' or '<module> <name>', got: {module_path}"
                )
            base_path, name = dotted.rsplit(".", 1)
        else:
            raise ValueError(
                f"spec must be a 2-tuple ('module', 'name') or a single dotted string, got: {module_path}"
            )
    elif isinstance(module_path, str):
        if "." not in module_path:
            raise ValueError(
                f"spec string must be of form 'module.name', got: {module_path}"
            )
        base_path, name = module_path.rsplit(".", 1)
    else:
        raise TypeError(
            f"module_path must be tuple, list, or str, got type {type(module_path)}"
        )

    try:
        module = __import__(base_path, globals(), locals(), [name])
    except ImportError as e:
        print(f"couldn't import module due to {e}")
        return None
    return vars(module)[name]


def get_module(spec_or_module: Union[ModuleSpec, type], **additional_kwargs):
    # If a module clas is already provided return it as is
    if isinstance(spec_or_module, (type, types.FunctionType)):
        return spec_or_module

    # If the module is provided instead of module path, then return it as is
    if isinstance(spec_or_module.module, (type, types.FunctionType)):
        return spec_or_module.module

    # Otherwise, return the dynamically imported module from the module path
    return import_module(spec_or_module.module)


def build_module(spec_or_module: Union[ModuleSpec, type], *args, **kwargs):
    # If the passed `spec_or_module` is
    # a `Function`, then return it as it is
    # NOTE: to support an already initialized module add the following condition
    # `or isinstance(spec_or_module, torch.nn.Module)` to the following if check
    if isinstance(spec_or_module, types.FunctionType):
        return spec_or_module

    # If the passed `spec_or_module` is actually a spec (instance of
    # `ModuleSpec`) and it specifies a `Function` using its `module`
    # field, return the `Function` as it is
    if isinstance(spec_or_module, ModuleSpec) and isinstance(
        spec_or_module.module, types.FunctionType
    ):
        return spec_or_module.module

    # Check if a module class is provided as a spec or if the module path
    # itself is a class
    if isinstance(spec_or_module, type):
        module = spec_or_module
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):
        module = spec_or_module.module
    else:
        # Otherwise, dynamically import the module from the module path
        module = import_module(spec_or_module.module)

    # If the imported module is actually a `Function` return it as it is
    if isinstance(module, types.FunctionType):
        return module

    # Finally return the initialized module with params from the spec as well
    # as those passed as **kwargs from the code

    # Add the `submodules` argument to the module init call if it exists in the
    # spec.
    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:
        kwargs["submodules"] = spec_or_module.submodules

    try:
        return module(
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs
        )
    except Exception as e:
        # improve the error message since we hide the module name in the line above
        import sys

        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )
