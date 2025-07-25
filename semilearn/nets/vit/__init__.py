# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .vit import vit_tiny_patch2_32, vit_small_patch2_32, vit_small_patch16_224, vit_base_patch16_224, vit_base_patch16_96
from .vit import VisionTransformer

from .vpt import vpt_on_vit_base_patch_16_224, ftcls_on_vit_base_patch_16_224
from .dualpt import dualpt_on_vit_base_patch_16_224
