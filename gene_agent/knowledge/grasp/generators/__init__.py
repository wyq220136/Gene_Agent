"""当前已迁移 generator 的显式注册表。"""
from .cylinder_axial import generate_cylinder_axial_local
from .ring import generate_ring_local
from .trifold_handle import generate_trifold_handle_local
from .cylinder import generate_cylinder_local
from .cuboidal_block import generate_cuboidal_block_local
GENERATOR_REGISTRY = {"ring": generate_ring_local, "cylinder_axial": generate_cylinder_axial_local, "trifold_handle": generate_trifold_handle_local, "cylinder": generate_cylinder_local, "cuboidal_block": generate_cuboidal_block_local}
__all__ = ["GENERATOR_REGISTRY", "generate_cylinder_axial_local", "generate_ring_local"]
