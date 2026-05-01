# backbone.py
"""
Backbone Model for PatchCore
==============================
This file loads the pretrained CNN that acts as our feature extractor.
We use WideResNet50 (trained on ImageNet) and freeze it completely —
no weights are updated. We only use it to "look" at images and grab
the internal representations from its middle layers.

Hooks: PyTorch lets you attach "hooks" to layers. A hook is a callback
that runs every time data flows through that layer, capturing the output.
We hook layer2 (fine detail) and layer3 (coarse shapes).

How it fits: This file provides the model to feature_extractor.py.
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


def load_backbone(
    device: str,
    backbone_name: str = "wide_resnet50_2",
) -> nn.Module:
    """Load a frozen, ImageNet-pretrained WideResNet50.

    The model is set to eval mode and all gradients are disabled so
    no memory is wasted on the computational graph.

    Args:
        device: 'cuda' or 'cpu'.
        backbone_name: torchvision model name (default 'wide_resnet50_2').

    Returns:
        The frozen model on the specified device.
    """
    model = getattr(models, backbone_name)(weights="IMAGENET1K_V1")
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        "Backbone loaded: %s (frozen, eval mode, device=%s)",
        backbone_name, device,
    )
    return model


def register_hooks(
    model: nn.Module,
    layer_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHook]]:
    """Attach forward hooks to capture intermediate feature maps.

    Each hook stores its layer's output in a shared dictionary every
    time a forward pass runs. The dictionary is keyed by layer name.

    Args:
        model: The backbone model.
        layer_names: Which layers to hook (e.g. ['layer2', 'layer3']).

    Returns:
        hook_dict: Dict that will be populated with feature maps after
                   each forward pass. Keys are layer names.
        handles: List of hook handles — pass to remove_hooks() when done.
    """
    hook_dict: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHook] = []

    for name in layer_names:
        layer = getattr(model, name, None)
        if layer is None:
            logger.error("Layer '%s' not found in model.", name)
            raise AttributeError(f"Layer '{name}' not found in model.")

        def _make_hook(layer_name: str):
            """Closure so each hook captures its own layer_name."""
            def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
                hook_dict[layer_name] = output.detach()
            return hook_fn

        handle = layer.register_forward_hook(_make_hook(name))
        handles.append(handle)
        logger.info("Hook registered on '%s'.", name)

    return hook_dict, handles


def remove_hooks(
    handles: List[torch.utils.hooks.RemovableHook],
) -> None:
    """Detach all hook handles to free memory.

    Call this when you're done extracting features to clean up.

    Args:
        handles: List of hook handles returned by register_hooks().
    """
    for handle in handles:
        handle.remove()
    logger.info("Removed %d hook(s).", len(handles))
