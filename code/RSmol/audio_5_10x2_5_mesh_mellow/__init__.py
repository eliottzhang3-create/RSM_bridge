"""Isolated ReasonAQA + HTSAT + MeSH audio training route."""

__all__ = ["AudioMeshModel", "AudioMeshConfig", "build_labels"]


def __getattr__(name: str):
    if name in __all__:
        from .model import AudioMeshModel, AudioMeshConfig, build_labels
        return {"AudioMeshModel": AudioMeshModel, "AudioMeshConfig": AudioMeshConfig, "build_labels": build_labels}[name]
    raise AttributeError(name)
