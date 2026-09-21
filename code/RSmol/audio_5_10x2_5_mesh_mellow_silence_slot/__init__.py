"""Isolated fixed-260 MeSH route with a runtime-silence second audio slot."""

__all__ = ["AudioMeshSilenceSlotConfig", "AudioMeshSilenceSlotModel"]


def __getattr__(name: str):
    if name in __all__:
        from .model import AudioMeshSilenceSlotConfig, AudioMeshSilenceSlotModel
        return {
            "AudioMeshSilenceSlotConfig": AudioMeshSilenceSlotConfig,
            "AudioMeshSilenceSlotModel": AudioMeshSilenceSlotModel,
        }[name]
    raise AttributeError(name)
