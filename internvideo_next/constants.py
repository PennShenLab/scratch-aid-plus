"""Project constants for binary mouse scratching classification."""

DEFAULT_MODEL_VARIANT = "base"
LEGACY_MEAN_POOLING = "mean"
PRETRAINED_PROJECTOR_POOLING = "pretrained_projector"
ATTENTIVE_PROBE_POOLING = "attentive_probe"
POOLING_MODES = (
    LEGACY_MEAN_POOLING,
    PRETRAINED_PROJECTOR_POOLING,
    ATTENTIVE_PROBE_POOLING,
)
CACHE_POOLING_MODES = (LEGACY_MEAN_POOLING, PRETRAINED_PROJECTOR_POOLING)
DEFAULT_POOLING_MODE = PRETRAINED_PROJECTOR_POOLING
FROZEN_TRAIN_MODE = "frozen"
FINETUNE_TRAIN_MODE = "finetune"
LORA_TRAIN_MODE = "lora"
TRAIN_MODES = (FROZEN_TRAIN_MODE, FINETUNE_TRAIN_MODE, LORA_TRAIN_MODE)
CACHED_FROZEN_MODE = "cached_frozen"
BENCHMARK_TRAIN_MODES = (*TRAIN_MODES, CACHED_FROZEN_MODE)
MODEL_VARIANTS = {
    "base": {
        "model_id": "revliter/internvideo_next_base_p14_res224_f16",
        "revision": "ff2659b9be360a6b1e94b1eb381778a960da6019",
        "scale": "smaller base variant",
        "description": "Default frozen-backbone choice for the current small mouse video dataset.",
    },
    "large": {
        "model_id": "revliter/internvideo_next_large_p14_res224_f16",
        "revision": "b14495944fe821fbc2fbdd7f01404508ba0d5a6f",
        "scale": "about 0.3B parameters",
        "description": "Retained for larger GPU-cluster runs and direct comparison with the original 0.3B setup.",
    },
}
MODEL_ID = MODEL_VARIANTS[DEFAULT_MODEL_VARIANT]["model_id"]
LABEL_NAMES = {0: "nonscratching", 1: "scratching"}
CLASS1_LABEL = 1
NUM_FRAMES = 16
IMAGE_SIZE = 224
GROUP_ID_SCHEMA = "dataset_namespace_plus_normalized_video_id_v1"


def resolve_model_choice(
    model_variant: str = DEFAULT_MODEL_VARIANT,
    model_id: str | None = None,
    model_revision: str | None = None,
) -> dict[str, str | None]:
    """Resolve a named model variant or a custom Hugging Face model id."""

    if model_id:
        for variant, spec in MODEL_VARIANTS.items():
            if spec["model_id"] == model_id:
                return {
                    "variant": variant,
                    **spec,
                    "revision": model_revision or spec["revision"],
                }
        return {
            "variant": "custom",
            "model_id": model_id,
            "revision": model_revision,
            "scale": "custom model",
            "description": "Custom Hugging Face model id supplied with --model-id.",
        }
    if model_variant not in MODEL_VARIANTS:
        choices = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(
            f"Unknown model variant {model_variant!r}. Choose one of: {choices}"
        )
    return {
        "variant": model_variant,
        **MODEL_VARIANTS[model_variant],
        "revision": model_revision or MODEL_VARIANTS[model_variant]["revision"],
    }
