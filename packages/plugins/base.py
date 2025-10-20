"""Plugin architecture for Summit.OS domain extensions."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
import importlib
import os


class PluginType(str, Enum):
    """Types of plugins supported by Summit.OS."""
    THREAT_ASSESSOR = "threat_assessor"
    PAYLOAD_HANDLER = "payload_handler"
    MISSION_PLANNER = "mission_planner"
    DATA_PROCESSOR = "data_processor"
    COMMUNICATION = "communication"


class PluginMetadata(BaseModel):
    """Metadata for a Summit.OS plugin."""
    name: str
    version: str
    description: str
    author: str
    domain: str
    plugin_type: PluginType
    dependencies: List[str] = Field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None


class BasePlugin(ABC):
    """Abstract base class for all Summit.OS plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._metadata = self.get_metadata()
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self._metadata
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of the plugin."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        # Override in subclasses for specific validation
        return True


class PluginRegistry:
    """Registry for managing Summit.OS plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugins_by_domain: Dict[str, List[BasePlugin]] = {}
        self._plugins_by_type: Dict[PluginType, List[BasePlugin]] = {}
    
    def register_plugin(self, plugin: BasePlugin) -> bool:
        """Register a plugin with the registry."""
        try:
            metadata = plugin.metadata
            plugin_key = f"{metadata.domain}:{metadata.name}"
            
            # Check for conflicts
            if plugin_key in self._plugins:
                raise ValueError(f"Plugin {plugin_key} already registered")
            
            # Register plugin
            self._plugins[plugin_key] = plugin
            
            # Add to domain index
            if metadata.domain not in self._plugins_by_domain:
                self._plugins_by_domain[metadata.domain] = []
            self._plugins_by_domain[metadata.domain].append(plugin)
            
            # Add to type index
            if metadata.plugin_type not in self._plugins_by_type:
                self._plugins_by_type[metadata.plugin_type] = []
            self._plugins_by_type[metadata.plugin_type].append(plugin)
            
            return True
            
        except Exception as e:
            print(f"Failed to register plugin: {e}")
            return False
    
    def get_plugin(self, domain: str, name: str) -> Optional[BasePlugin]:
        """Get a specific plugin by domain and name."""
        plugin_key = f"{domain}:{name}"
        return self._plugins.get(plugin_key)
    
    def get_plugins_by_domain(self, domain: str) -> List[BasePlugin]:
        """Get all plugins for a specific domain."""
        return self._plugins_by_domain.get(domain, [])
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all plugins of a specific type."""
        return self._plugins_by_type.get(plugin_type, [])
    
    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return list(self._plugins_by_domain.keys())
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List metadata for all registered plugins."""
        return [plugin.metadata for plugin in self._plugins.values()]
    
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered plugins."""
        results = {}
        for key, plugin in self._plugins.items():
            try:
                success = await plugin.initialize()
                results[key] = success
                if not success:
                    print(f"Failed to initialize plugin: {key}")
            except Exception as e:
                print(f"Error initializing plugin {key}: {e}")
                results[key] = False
        return results
    
    async def shutdown_all(self) -> None:
        """Shutdown all registered plugins."""
        for plugin in self._plugins.values():
            try:
                await plugin.shutdown()
            except Exception as e:
                print(f"Error shutting down plugin: {e}")


class PluginLoader:
    """Dynamic plugin loader for Summit.OS."""
    
    def __init__(self, plugin_registry: PluginRegistry):
        self.registry = plugin_registry
    
    def load_plugins_from_directory(self, directory: str) -> Dict[str, bool]:
        """Load all plugins from a directory."""
        results = {}
        
        if not os.path.exists(directory):
            return results
        
        for item in os.listdir(directory):
            plugin_dir = os.path.join(directory, item)
            if os.path.isdir(plugin_dir):
                result = self.load_plugin_from_directory(plugin_dir)
                results[item] = result
        
        return results
    
    def load_plugin_from_directory(self, plugin_dir: str) -> bool:
        """Load a single plugin from a directory."""
        try:
            # Look for plugin.py or __init__.py
            plugin_file = None
            if os.path.exists(os.path.join(plugin_dir, "plugin.py")):
                plugin_file = "plugin"
            elif os.path.exists(os.path.join(plugin_dir, "__init__.py")):
                plugin_file = "__init__"
            
            if not plugin_file:
                return False
            
            # Import the plugin module
            plugin_name = os.path.basename(plugin_dir)
            spec = importlib.util.spec_from_file_location(
                plugin_name,
                os.path.join(plugin_dir, f"{plugin_file}.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin class (should inherit from BasePlugin)
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    plugin_class = attr
                    break
            
            if not plugin_class:
                return False
            
            # Instantiate and register plugin
            plugin_instance = plugin_class()
            return self.registry.register_plugin(plugin_instance)
            
        except Exception as e:
            print(f"Error loading plugin from {plugin_dir}: {e}")
            return False


# Global plugin registry
plugin_registry = PluginRegistry()
plugin_loader = PluginLoader(plugin_registry)