from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_wan_package(root_dir: Path, version: str) -> ModuleType:
    """
    Load third_party/wanXX/wan as a uniquely named python package:

      - version="2.1" -> package name "wan_v21"
      - version="2.2" -> package name "wan_v22"

    This avoids sys.path/sys.modules conflicts between different WAN versions.
    """
    if version == "2.1":
        pkg_dir = root_dir / "third_party" / "wan21" / "wan"
        pkg_name = "wan_v21"
    elif version == "2.2":
        pkg_dir = root_dir / "third_party" / "wan22" / "wan"
        pkg_name = "wan_v22"
    else:
        raise ValueError(f"Unknown wan version: {version}")

    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"Cannot find {init_py}")

    # If already loaded, reuse it.
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]  # type: ignore[return-value]

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create spec for {pkg_name} from {init_py}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod

