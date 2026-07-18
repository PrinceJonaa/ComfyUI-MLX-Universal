if __name__ == "__main__":
    import sys

    print(
        "This file is a ComfyUI custom node package entry point and cannot be run directly as a script."
    )
    print(
        "To use these nodes, install this directory into your ComfyUI 'custom_nodes' directory and run ComfyUI."
    )
    sys.exit(0)

try:
    import sentry_sdk

    sentry_sdk.init(
        dsn="https://a2a24e1fc2f7fafe6aca650ca28b777f@o4511620719443968.ingest.us.sentry.io/4511620737073152",
        send_default_pii=True,
        enable_logs=True,
        traces_sample_rate=1.0,
        profile_session_sample_rate=1.0,
    )
except Exception as e:
    print(f"[ComfyUI-MLX-Universal] Warning: Failed to initialize Sentry SDK: {e}")


from .nodes.audio_nodes import (
    NODE_CLASS_MAPPINGS as AUDIO_MAPPINGS,
)
from .nodes.audio_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as AUDIO_DISPLAY,
)
from .nodes.batch_nodes import (
    NODE_CLASS_MAPPINGS as BATCH_MAPPINGS,
)
from .nodes.batch_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as BATCH_DISPLAY,
)
from .nodes.diffusion_nodes import (
    NODE_CLASS_MAPPINGS as DIFFUSION_MAPPINGS,
)
from .nodes.diffusion_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as DIFFUSION_DISPLAY,
)
from .nodes.embedding_nodes import (
    NODE_CLASS_MAPPINGS as EMBEDDING_MAPPINGS,
)
from .nodes.embedding_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as EMBEDDING_DISPLAY,
)
from .nodes.generate_nodes import (
    NODE_CLASS_MAPPINGS as GENERATE_MAPPINGS,
)
from .nodes.generate_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as GENERATE_DISPLAY,
)
from .nodes.loader_nodes import (
    NODE_CLASS_MAPPINGS as LOADER_MAPPINGS,
)
from .nodes.loader_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY,
)
from .nodes.sam_nodes import (
    NODE_CLASS_MAPPINGS as SAM_MAPPINGS,
)
from .nodes.sam_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as SAM_DISPLAY,
)
from .nodes.system_nodes import (
    NODE_CLASS_MAPPINGS as SYSTEM_MAPPINGS,
)
from .nodes.system_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as SYSTEM_DISPLAY,
)
from .nodes.video_nodes import (
    NODE_CLASS_MAPPINGS as VIDEO_MAPPINGS,
)
from .nodes.video_nodes import (
    NODE_DISPLAY_NAME_MAPPINGS as VIDEO_DISPLAY,
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(LOADER_MAPPINGS)
NODE_CLASS_MAPPINGS.update(GENERATE_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SAM_MAPPINGS)
NODE_CLASS_MAPPINGS.update(VIDEO_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SYSTEM_MAPPINGS)
NODE_CLASS_MAPPINGS.update(DIFFUSION_MAPPINGS)
NODE_CLASS_MAPPINGS.update(AUDIO_MAPPINGS)
NODE_CLASS_MAPPINGS.update(EMBEDDING_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BATCH_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(LOADER_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(GENERATE_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(SAM_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(SYSTEM_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(DIFFUSION_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(AUDIO_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(EMBEDDING_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(BATCH_DISPLAY)
