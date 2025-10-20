"""Plugin system for Summit.OS domain extensions."""

from .base import (
    PluginType,
    PluginMetadata,
    BasePlugin,
    PluginRegistry,
    PluginLoader,
    plugin_registry,
    plugin_loader
)

__all__ = [
    "PluginType",
    "PluginMetadata",
    "BasePlugin", 
    "PluginRegistry",
    "PluginLoader",
    "plugin_registry",
    "plugin_loader"
]