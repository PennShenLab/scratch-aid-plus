"""InternVideo-Next frozen backbone and binary classifier head."""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .constants import (
    ATTENTIVE_PROBE_POOLING,
    DEFAULT_POOLING_MODE,
    FINETUNE_TRAIN_MODE,
    FROZEN_TRAIN_MODE,
    LEGACY_MEAN_POOLING,
    LORA_TRAIN_MODE,
    MODEL_ID,
    MODEL_VARIANTS,
    POOLING_MODES,
    PRETRAINED_PROJECTOR_POOLING,
    TRAIN_MODES,
)


POOLING_DESCRIPTIONS = {
    LEGACY_MEAN_POOLING: "legacy unweighted mean over raw backbone tokens",
    PRETRAINED_PROJECTOR_POOLING: "Hugging Face InternVideo-Next pretrained CLIP projector",
    ATTENTIVE_PROBE_POOLING: "trainable single-layer cross-attention probe over raw backbone tokens",
}

LORA_CHECKPOINT_TYPE = "lora_adapter"
LORA_SUPPORTED_TARGETS = ("qkv", "proj")


class LoRALinear(nn.Module):
    """A frozen linear layer with an additive, FP32 LoRA update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / rank
        self.lora_dropout = nn.Dropout(dropout)
        factory_kwargs = {"device": base_layer.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features, **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.empty(base_layer.out_features, rank, **factory_kwargs)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        adapter_inputs = self.lora_dropout(inputs).to(self.lora_A.dtype)
        adapter_output = F.linear(F.linear(adapter_inputs, self.lora_A), self.lora_B)
        return base_output + (adapter_output * self.scaling).to(base_output.dtype)


def checkpoint_type_for_train_mode(train_mode: str) -> str:
    if train_mode == FROZEN_TRAIN_MODE:
        return "classifier_head"
    if train_mode == FINETUNE_TRAIN_MODE:
        return "full_model"
    if train_mode == LORA_TRAIN_MODE:
        return LORA_CHECKPOINT_TYPE
    raise ValueError(f"Unsupported training mode: {train_mode!r}")


def _internvideo_transformer_blocks(backbone: nn.Module) -> nn.ModuleList:
    """Resolve the pinned HF InternVideo-Next transformer block path fail-closed."""

    inner = getattr(backbone, "model", None)
    blocks = getattr(inner, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise RuntimeError(
            "Expected pinned InternVideo-Next transformer blocks at backbone.model.blocks"
        )
    if not blocks:
        raise RuntimeError("InternVideo-Next backbone.model.blocks is empty")
    return blocks


def configure_lora_adapters(
    model: "InternVideoNextBinaryClassifier",
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    last_n_blocks: int = 4,
    target_modules: tuple[str, ...] = LORA_SUPPORTED_TARGETS,
) -> dict[str, Any]:
    """Inject LoRA into an exact suffix of attention qkv/proj modules."""

    if model.train_mode != LORA_TRAIN_MODE:
        raise ValueError("LoRA adapters require train_mode='lora'")
    if last_n_blocks < 1:
        raise ValueError("LoRA last_n_blocks must be at least 1")
    if not target_modules or len(set(target_modules)) != len(target_modules):
        raise ValueError("LoRA target_modules must be non-empty and unique")
    unsupported = sorted(set(target_modules).difference(LORA_SUPPORTED_TARGETS))
    if unsupported:
        raise ValueError(f"Unsupported LoRA target modules: {unsupported}")
    if getattr(model, "lora_config", None) is not None:
        raise RuntimeError("LoRA adapters are already configured")

    model.backbone.requires_grad_(False)
    blocks = _internvideo_transformer_blocks(model.backbone)
    depth = len(blocks)
    if last_n_blocks > depth:
        raise ValueError(
            f"Requested the last {last_n_blocks} LoRA blocks, but backbone depth is {depth}"
        )

    target_names: list[str] = []
    hidden_widths: set[int] = set()
    start_index = depth - last_n_blocks
    for block_index in range(start_index, depth):
        attention = getattr(blocks[block_index], "attn", None)
        if not isinstance(attention, nn.Module):
            raise RuntimeError(
                f"Expected attention module at backbone.model.blocks.{block_index}.attn"
            )
        for target_name in target_modules:
            linear = getattr(attention, target_name, None)
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(
                    "Expected nn.Linear at "
                    f"backbone.model.blocks.{block_index}.attn.{target_name}; "
                    f"found {type(linear)!r}"
                )
            hidden_widths.add(linear.in_features)
            setattr(
                attention,
                target_name,
                LoRALinear(
                    linear,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            target_names.append(f"model.blocks.{block_index}.attn.{target_name}")

    expected_target_count = last_n_blocks * len(target_modules)
    if len(target_names) != expected_target_count:
        raise RuntimeError(
            f"Injected {len(target_names)} LoRA modules; expected {expected_target_count}"
        )
    adapter_parameter_count = sum(
        parameter.numel() for parameter in model.lora_parameters()
    )
    config = {
        "rank": rank,
        "alpha": float(alpha),
        "dropout": float(dropout),
        "last_n_blocks": last_n_blocks,
        "target_modules": list(target_modules),
        "backbone_depth": depth,
        "hidden_widths": sorted(hidden_widths),
        "target_module_count": len(target_names),
        "adapter_parameter_count": adapter_parameter_count,
    }
    model.lora_config = config
    model.lora_target_module_names = target_names
    return config


def lora_adapter_state_dict(
    model: "InternVideoNextBinaryClassifier",
) -> dict[str, torch.Tensor]:
    """Return only LoRA trainable tensors, never frozen backbone weights."""

    if model.train_mode != LORA_TRAIN_MODE or model.lora_config is None:
        raise RuntimeError("LoRA adapters are not configured")
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.backbone.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    }
    expected_tensors = 2 * len(model.lora_target_module_names)
    if len(state) != expected_tensors:
        raise RuntimeError(
            f"Found {len(state)} LoRA tensors; expected {expected_tensors}"
        )
    return state


def load_lora_adapter_state_dict(
    model: "InternVideoNextBinaryClassifier",
    state: Mapping[str, torch.Tensor],
) -> None:
    """Load an adapter-only state dict with exact key and shape validation."""

    current = {
        name: parameter
        for name, parameter in model.backbone.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    }
    if set(state) != set(current):
        missing = sorted(set(current).difference(state))
        unexpected = sorted(set(state).difference(current))
        raise RuntimeError(
            f"LoRA state keys differ; missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            source = state[name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"LoRA tensor {name} shape {tuple(source.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))


def infer_feature_dim(config: Any, default: int = 1024) -> int:
    """Infer the feature width from a remote-code config."""

    for attr in ("hidden_size", "embed_dim", "vision_width", "num_features"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    model_config = getattr(config, "model_config", None)
    if isinstance(model_config, dict):
        for attr in ("hidden_size", "embed_dim", "vision_width", "num_features"):
            value = model_config.get(attr)
            if isinstance(value, int) and value > 0:
                return value
    elif model_config is not None:
        for attr in ("hidden_size", "embed_dim", "vision_width", "num_features"):
            value = getattr(model_config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return default


def infer_projected_feature_dim(config: Any, default: int = 768) -> int:
    """Infer the pretrained projector width from an InternVideo-Next config."""

    containers = [config]
    for attr in ("model_config", "vision_config"):
        value = getattr(config, attr, None)
        if value is not None:
            containers.append(value)
    for container in containers:
        for attr in ("clip_embed_dim", "projection_dim", "projector_dim"):
            value = (
                container.get(attr)
                if isinstance(container, dict)
                else getattr(container, attr, None)
            )
            if isinstance(value, int) and value > 0:
                return value
    return default


def infer_attention_pool_num_heads(config: Any, feature_dim: int) -> int:
    """Infer the official attention-pooling head count with safe custom fallbacks."""

    containers = [config]
    for attr in ("model_config", "vision_config"):
        value = getattr(config, attr, None)
        if value is not None:
            containers.append(value)
    for container in containers:
        value = (
            container.get("attn_pool_num_heads")
            if isinstance(container, dict)
            else getattr(container, "attn_pool_num_heads", None)
        )
        if isinstance(value, int) and value > 0 and feature_dim % value == 0:
            return value
    for candidate in (16, 12, 8, 4, 1):
        if feature_dim % candidate == 0:
            return candidate
    return 1


def _default_revision_for_model(model_id: str) -> str | None:
    for spec in MODEL_VARIANTS.values():
        if spec["model_id"] == model_id:
            return str(spec["revision"])
    return None


def _first_feature_tensor(output: Any) -> torch.Tensor:
    """Unwrap common Hugging Face output containers without tensor truth tests."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        if not output:
            raise TypeError("Backbone returned an empty feature sequence")
        return _first_feature_tensor(output[0])

    feature_keys = (
        "projected_features",
        "video_embeds",
        "video_features",
        "pooler_output",
        "last_hidden_state",
        "features",
    )
    if isinstance(output, Mapping):
        for key in feature_keys:
            value = output.get(key)
            if value is not None:
                return _first_feature_tensor(value)
        if len(output) == 1:
            return _first_feature_tensor(next(iter(output.values())))
    else:
        for key in feature_keys:
            value = getattr(output, key, None)
            if value is not None:
                return _first_feature_tensor(value)
    raise TypeError(f"Unsupported backbone feature output type: {type(output)!r}")


def _as_feature_tokens(features: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Convert token-last or channel-first backbone features to B,N,D."""

    if features.ndim < 2:
        raise ValueError(f"Unsupported feature tensor shape: {tuple(features.shape)}")
    if features.ndim == 2:
        if features.shape[-1] != feature_dim:
            raise ValueError(
                f"Expected feature width {feature_dim}, got tensor shape {tuple(features.shape)}"
            )
        return features.unsqueeze(1)
    if features.shape[-1] == feature_dim:
        return features.reshape(features.shape[0], -1, feature_dim)
    if features.shape[1] == feature_dim:
        return features.movedim(1, -1).reshape(features.shape[0], -1, feature_dim)
    raise ValueError(
        f"Cannot locate expected feature width {feature_dim} in tensor shape {tuple(features.shape)}"
    )


def _mean_pool_features(features: torch.Tensor, feature_dim: int) -> torch.Tensor:
    return _as_feature_tokens(features, feature_dim).mean(dim=1)


class AttentionProbe(nn.Module):
    """Trainable one-layer attention pooler matching the official block shape."""

    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(feature_dim, eps=1e-5)
        self.norm_k = nn.LayerNorm(feature_dim, eps=1e-5)
        self.norm_v = nn.LayerNorm(feature_dim, eps=1e-5)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        nn.init.trunc_normal_(self.cross_attention.in_proj_weight, std=0.02)
        nn.init.zeros_(self.cross_attention.in_proj_bias)
        nn.init.trunc_normal_(self.cross_attention.out_proj.weight, std=0.02)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = _as_feature_tokens(features.float(), self.feature_dim)
        query = tokens.mean(dim=1, keepdim=True)
        pooled, _weights = self.cross_attention(
            self.norm_q(query),
            self.norm_k(tokens),
            self.norm_v(tokens),
            need_weights=False,
        )
        return pooled.squeeze(1)


def _set_model_config_value(config: Any, key: str, value: Any) -> None:
    model_config = getattr(config, "model_config", None)
    if model_config is None:
        model_config = {}
        setattr(config, "model_config", model_config)
    if isinstance(model_config, dict):
        model_config[key] = value
    else:
        setattr(model_config, key, value)


def disable_flash_extensions(config: Any) -> None:
    """Force the remote InternVideo code onto pure PyTorch execution paths."""

    for key in ("use_flash_attn", "use_fused_rmsnorm", "use_fused_mlp"):
        _set_model_config_value(config, key, False)


def install_flash_attn_import_stubs() -> None:
    """Let HF remote code import optional flash-attn symbols that stay unused."""

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "FlashAttention is disabled for this frozen-backbone pipeline"
        )

    class UnavailableModule(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            raise RuntimeError("Fused FlashAttention modules are disabled")

    root = types.ModuleType("flash_attn")
    root.__path__ = []
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.flash_attn_varlen_qkvpacked_func = unavailable
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.unpad_input = unavailable
    bert_padding.pad_input = unavailable
    modules = types.ModuleType("flash_attn.modules")
    modules.__path__ = []
    mlp = types.ModuleType("flash_attn.modules.mlp")
    mlp.FusedMLP = UnavailableModule
    ops = types.ModuleType("flash_attn.ops")
    ops.__path__ = []
    rms_norm = types.ModuleType("flash_attn.ops.rms_norm")
    rms_norm.DropoutAddRMSNorm = UnavailableModule

    sys.modules.update(
        {
            "flash_attn": root,
            "flash_attn.flash_attn_interface": interface,
            "flash_attn.bert_padding": bert_padding,
            "flash_attn.modules": modules,
            "flash_attn.modules.mlp": mlp,
            "flash_attn.ops": ops,
            "flash_attn.ops.rms_norm": rms_norm,
        }
    )


class InternVideoNextBinaryClassifier(nn.Module):
    """InternVideo-Next backbone plus a lightweight binary classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        dropout: float,
        freeze_backbone: bool = True,
        model_id: str = MODEL_ID,
        pooling_mode: str = DEFAULT_POOLING_MODE,
        backbone_feature_dim: int | None = None,
        model_revision: str | None = None,
        resolved_model_revision: str | None = None,
        loading_info: dict[str, Any] | None = None,
        attention_probe_num_heads: int = 16,
        train_mode: str | None = None,
    ) -> None:
        super().__init__()
        if pooling_mode not in POOLING_MODES:
            choices = ", ".join(POOLING_MODES)
            raise ValueError(
                f"Unknown pooling mode {pooling_mode!r}. Choose one of: {choices}"
            )
        resolved_train_mode = train_mode or (
            FROZEN_TRAIN_MODE if freeze_backbone else FINETUNE_TRAIN_MODE
        )
        if resolved_train_mode not in TRAIN_MODES:
            raise ValueError(f"Unsupported training mode: {resolved_train_mode!r}")
        if resolved_train_mode == FINETUNE_TRAIN_MODE and freeze_backbone:
            raise ValueError("train_mode='finetune' cannot freeze the backbone")
        if (
            resolved_train_mode in (FROZEN_TRAIN_MODE, LORA_TRAIN_MODE)
            and not freeze_backbone
        ):
            raise ValueError(
                f"train_mode={resolved_train_mode!r} must freeze base backbone weights"
            )
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.backbone_feature_dim = backbone_feature_dim or feature_dim
        self.dropout = dropout
        self.freeze_backbone = freeze_backbone
        self.train_mode = resolved_train_mode
        self.model_id = model_id
        self.model_revision = model_revision
        self.resolved_model_revision = resolved_model_revision
        self.loading_info = loading_info or {}
        self.attention_probe_num_heads = attention_probe_num_heads
        self.pooling_mode = pooling_mode
        self.pooling_description = POOLING_DESCRIPTIONS[pooling_mode]
        self.lora_config: dict[str, Any] | None = None
        self.lora_target_module_names: list[str] = []
        self.attention_probe = (
            AttentionProbe(self.backbone_feature_dim, attention_probe_num_heads)
            if pooling_mode == ATTENTIVE_PROBE_POOLING
            else None
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2),
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = MODEL_ID,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        pooling_mode: str = DEFAULT_POOLING_MODE,
        revision: str | None = None,
        attention_probe_num_heads: int | None = None,
        train_mode: str | None = None,
    ) -> "InternVideoNextBinaryClassifier":
        from transformers import AutoConfig, AutoModel

        if pooling_mode not in POOLING_MODES:
            choices = ", ".join(POOLING_MODES)
            raise ValueError(
                f"Unknown pooling mode {pooling_mode!r}. Choose one of: {choices}"
            )
        effective_revision = revision or _default_revision_for_model(model_id)
        pretrained_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if effective_revision is not None:
            pretrained_kwargs["revision"] = effective_revision
        config = AutoConfig.from_pretrained(model_id, **pretrained_kwargs)
        disable_flash_extensions(config)
        install_flash_attn_import_stubs()
        loaded = AutoModel.from_pretrained(
            model_id,
            config=config,
            output_loading_info=True,
            **pretrained_kwargs,
        )
        if isinstance(loaded, tuple) and len(loaded) == 2:
            backbone, loading_info = loaded
        else:  # Defensive compatibility with non-standard remote wrappers.
            backbone, loading_info = loaded, {}
        backbone_feature_dim = infer_feature_dim(config)
        projected_feature_dim = infer_projected_feature_dim(config)
        resolved_probe_num_heads = (
            attention_probe_num_heads
            or infer_attention_pool_num_heads(
                config,
                backbone_feature_dim,
            )
        )
        feature_dim = (
            projected_feature_dim
            if pooling_mode == PRETRAINED_PROJECTOR_POOLING
            else backbone_feature_dim
        )
        resolved_revision = getattr(config, "_commit_hash", None)
        if resolved_revision is None:
            resolved_revision = getattr(
                getattr(backbone, "config", None), "_commit_hash", None
            )
        resolved_revision = resolved_revision or effective_revision
        return cls(
            backbone=backbone,
            feature_dim=feature_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            model_id=model_id,
            pooling_mode=pooling_mode,
            backbone_feature_dim=backbone_feature_dim,
            model_revision=effective_revision,
            resolved_model_revision=resolved_revision,
            loading_info=loading_info,
            attention_probe_num_heads=resolved_probe_num_heads,
            train_mode=train_mode,
        )

    @staticmethod
    def _prepare_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
        # The official model card uses B,T,C,H,W from the processor, then
        # permutes to B,C,T,H,W before calling extract_features.
        if (
            pixel_values.ndim == 5
            and pixel_values.shape[1] != 3
            and pixel_values.shape[2] == 3
        ):
            return pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        return pixel_values

    def extract_backbone_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_pixel_values(pixel_values)

        if hasattr(self.backbone, "extract_features"):
            features = self.backbone.extract_features(pixel_values=pixel_values)
        else:
            features = self.backbone(pixel_values=pixel_values)
        return _first_feature_tensor(features)

    def extract_projected_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Call the remote model's pretrained projector with signature fallback."""

        pixel_values = self._prepare_pixel_values(pixel_values)
        try:
            projected = self.backbone(pixel_values=pixel_values, projected=True)
        except TypeError as keyword_error:
            try:
                projected = self.backbone(pixel_values, projected=True)
            except TypeError:
                try:
                    # The published HF wrapper exposes only forward(pixel_values)
                    # and enables projected=True inside that method.
                    projected = self.backbone(pixel_values=pixel_values)
                except TypeError:
                    raise keyword_error
        return _first_feature_tensor(projected)

    def _validate_pooled_features(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2 or pooled.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Pooling mode {self.pooling_mode!r} expected B,{self.feature_dim} output, "
                f"got {tuple(pooled.shape)}"
            )
        return pooled

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Retain legacy raw-token mean pooling for old checkpoints and caches."""

        if self.pooling_mode != LEGACY_MEAN_POOLING:
            raise ValueError(
                f"pool_features(raw_features) is only valid for {LEGACY_MEAN_POOLING!r}; "
                f"pooling mode {self.pooling_mode!r} must use extract_pooled_features(pixel_values)"
            )
        return self._validate_pooled_features(
            _mean_pool_features(features, self.backbone_feature_dim)
        )

    def _attention_pool_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.attention_probe is None:
            raise RuntimeError(
                "Attention probe is unavailable for the selected pooling mode"
            )
        return self._validate_pooled_features(self.attention_probe(features))

    def extract_pooled_features(
        self,
        pixel_values: torch.Tensor,
        *,
        allow_trainable_pooling: bool = False,
    ) -> torch.Tensor:
        """Return B,D pooled features while guarding invalid attentive caches."""

        if self.pooling_mode == PRETRAINED_PROJECTOR_POOLING:
            projected = self.extract_projected_features(pixel_values)
            pooled = _mean_pool_features(projected, self.feature_dim)
            return self._validate_pooled_features(pooled)
        if self.pooling_mode == LEGACY_MEAN_POOLING:
            return self.pool_features(self.extract_backbone_features(pixel_values))
        if not allow_trainable_pooling:
            raise ValueError(
                "attentive_probe is trainable and cannot be stored as a deterministic frozen "
                "feature cache; use mean or pretrained_projector for cached_frozen"
            )
        return self._attention_pool_features(
            self.extract_backbone_features(pixel_values)
        )

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """Yield every lightweight trainable-head parameter, including the probe."""

        if self.attention_probe is not None:
            yield from self.attention_probe.parameters()
        yield from self.classifier.parameters()

    def trainable_head_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only currently trainable lightweight-head parameters."""

        yield from (
            parameter for parameter in self.head_parameters() if parameter.requires_grad
        )

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        """Yield adapter matrices in deterministic backbone traversal order."""

        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                yield module.lora_A
                yield module.lora_B

    def train(self, mode: bool = True) -> "InternVideoNextBinaryClassifier":
        """Keep the frozen base in eval mode while enabling LoRA dropout."""

        super().train(mode)
        if mode and self.train_mode in (FROZEN_TRAIN_MODE, LORA_TRAIN_MODE):
            self.backbone.eval()
            if self.train_mode == LORA_TRAIN_MODE:
                for module in self.backbone.modules():
                    if isinstance(module, LoRALinear):
                        module.train(True)
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.train_mode == FROZEN_TRAIN_MODE:
            self.backbone.eval()
            if self.pooling_mode == ATTENTIVE_PROBE_POOLING:
                with torch.no_grad():
                    features = self.extract_backbone_features(pixel_values)
                pooled = self._attention_pool_features(features)
            else:
                with torch.no_grad():
                    pooled = self.extract_pooled_features(pixel_values)
        else:
            pooled = self.extract_pooled_features(
                pixel_values,
                allow_trainable_pooling=self.pooling_mode == ATTENTIVE_PROBE_POOLING,
            )
        return self.classifier(pooled.float())


def set_backbone_dtype(
    model: InternVideoNextBinaryClassifier, dtype_name: str, device: torch.device
) -> None:
    """Optionally cast the frozen backbone for CUDA memory efficiency."""

    if dtype_name == "auto":
        dtype_name = "fp16" if device.type == "cuda" else "fp32"
    if dtype_name == "fp16":
        model.backbone.to(dtype=torch.float16)
    elif dtype_name == "bf16":
        model.backbone.to(dtype=torch.bfloat16)
    elif dtype_name == "fp32":
        model.backbone.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown backbone dtype: {dtype_name}")


def checkpoint_payload(
    model: InternVideoNextBinaryClassifier,
    train_args: dict[str, Any],
    epoch: int,
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a head-only, adapter-only, or full fine-tuning checkpoint."""

    payload = {
        "checkpoint_type": checkpoint_type_for_train_mode(model.train_mode),
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "resolved_model_revision": model.resolved_model_revision,
        "pooling_mode": model.pooling_mode,
        "pooling_description": model.pooling_description,
        "feature_dim": model.feature_dim,
        "backbone_feature_dim": model.backbone_feature_dim,
        "attention_probe_num_heads": model.attention_probe_num_heads,
        "dropout": model.dropout,
        "freeze_backbone": model.freeze_backbone,
        "train_mode": model.train_mode,
        "classifier_state_dict": model.classifier.state_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "train_args": train_args,
    }
    if model.attention_probe is not None:
        payload["attention_probe_state_dict"] = model.attention_probe.state_dict()
    if model.train_mode == LORA_TRAIN_MODE:
        if model.lora_config is None:
            raise RuntimeError(
                "Cannot save a LoRA checkpoint before adapter configuration"
            )
        payload.update(
            {
                "lora_config": model.lora_config,
                "lora_target_module_names": model.lora_target_module_names,
                "lora_state_dict": lora_adapter_state_dict(model),
            }
        )
    elif model.train_mode == FINETUNE_TRAIN_MODE:
        payload["model_state_dict"] = model.state_dict()
    return payload


def load_classifier_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    backbone_dtype: str = "auto",
    model_id_override: str | None = None,
    model_revision_override: str | None = None,
) -> InternVideoNextBinaryClassifier:
    """Load a trained frozen-head, LoRA-adapter, or full fine-tuned checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_mode = str(
        checkpoint.get(
            "train_mode",
            (
                FROZEN_TRAIN_MODE
                if checkpoint.get("freeze_backbone", True)
                else FINETUNE_TRAIN_MODE
            ),
        )
    )
    expected_checkpoint_type = checkpoint_type_for_train_mode(train_mode)
    observed_checkpoint_type = checkpoint.get("checkpoint_type")
    if (
        observed_checkpoint_type is not None
        and observed_checkpoint_type != expected_checkpoint_type
    ):
        raise ValueError(
            f"Checkpoint type {observed_checkpoint_type!r} does not match "
            f"train mode {train_mode!r}"
        )
    freeze_backbone = train_mode != FINETUNE_TRAIN_MODE
    pooling_mode = str(checkpoint.get("pooling_mode", LEGACY_MEAN_POOLING))
    revision = (
        model_revision_override
        or checkpoint.get("resolved_model_revision")
        or checkpoint.get("model_revision")
    )
    model = InternVideoNextBinaryClassifier.from_pretrained(
        model_id=model_id_override or checkpoint.get("model_id", MODEL_ID),
        dropout=float(checkpoint.get("dropout", 0.2)),
        freeze_backbone=freeze_backbone,
        pooling_mode=pooling_mode,
        revision=revision,
        attention_probe_num_heads=checkpoint.get("attention_probe_num_heads"),
        train_mode=train_mode,
    )
    checkpoint_feature_dim = checkpoint.get("feature_dim")
    if (
        checkpoint_feature_dim is not None
        and int(checkpoint_feature_dim) != model.feature_dim
    ):
        raise ValueError(
            f"Checkpoint feature width {checkpoint_feature_dim} does not match "
            f"{pooling_mode!r} model width {model.feature_dim}"
        )
    model.to(device)
    set_backbone_dtype(model, backbone_dtype, device)
    if train_mode == LORA_TRAIN_MODE:
        lora_config = checkpoint.get("lora_config")
        if not isinstance(lora_config, dict):
            raise KeyError("LoRA checkpoint is missing lora_config")
        configure_lora_adapters(
            model,
            rank=int(lora_config["rank"]),
            alpha=float(lora_config["alpha"]),
            dropout=float(lora_config["dropout"]),
            last_n_blocks=int(lora_config["last_n_blocks"]),
            target_modules=tuple(lora_config["target_modules"]),
        )
        lora_state = checkpoint.get("lora_state_dict")
        if not isinstance(lora_state, Mapping):
            raise KeyError("LoRA checkpoint is missing lora_state_dict")
        load_lora_adapter_state_dict(model, lora_state)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if train_mode != FINETUNE_TRAIN_MODE:
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        if model.attention_probe is not None:
            probe_state = checkpoint.get("attention_probe_state_dict")
            if probe_state is None:
                raise KeyError(
                    "attentive_probe checkpoint is missing attention_probe_state_dict"
                )
            model.attention_probe.load_state_dict(probe_state)
    model.eval()
    return model
