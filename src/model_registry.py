"""
Model Registry - Dynamic model fetching from Backboard.io API
Fetches available models and providers directly from the Backboard API.
"""

import asyncio
import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class ModelInfo:
    """Information about a specific model."""
    provider: str
    model_name: str
    display_name: str
    description: str = ""
    context_window: int = 0
    recommended_for: List[str] = field(default_factory=list)
    is_multimodal: bool = False
    supports_tools: bool = True
    cost_tier: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_id(self) -> str:
        """Full identifier for the model."""
        return f"{self.provider}/{self.model_name}"
    
    @property
    def search_text(self) -> str:
        """Text used for search matching."""
        return f"{self.provider} {self.model_name} {self.display_name} {self.description} {' '.join(self.recommended_for)}".lower()


class BackboardModelRegistry:
    """Fetches and manages models from Backboard API."""
    
    BASE_URL = "https://app.backboard.io/api"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._models_cache: Optional[List[ModelInfo]] = None
        self._providers_cache: Optional[List[str]] = None
    
    async def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make an async request to Backboard API."""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        headers = {"X-API-Key": self.api_key}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                print(f"Error fetching {endpoint}: {e}")
                return {}
    
    async def fetch_providers(self) -> List[str]:
        """Fetch all available LLM providers from Backboard API."""
        if self._providers_cache:
            return self._providers_cache
        
        data = await self._make_request("providers")
        providers = data.get("providers", [])
        
        # Extract provider names
        if isinstance(providers, list):
            if providers and isinstance(providers[0], dict):
                self._providers_cache = [p.get("name", p.get("id", "")) for p in providers]
            else:
                self._providers_cache = providers
        
        return self._providers_cache or []
    
    async def fetch_models(self) -> List[ModelInfo]:
        """Fetch all available models from Backboard API."""
        if self._models_cache:
            return self._models_cache
        
        # Fetch with pagination to get all models
        all_models = []
        page = 1
        per_page = 100
        
        while True:
            # Try with pagination parameters
            data = await self._make_request(f"models?page={page}&per_page={per_page}")
            
            # Also try with limit parameter as fallback
            if not data.get("models"):
                data = await self._make_request(f"models?limit=10000")
            
            models_data = data.get("models", [])
            
            if not models_data:
                break
            
            for model_data in models_data:
                # Parse model information from API response
                provider = model_data.get("provider", "unknown")
                model_name = model_data.get("name", model_data.get("id", ""))
                
                # Infer display name
                display_name = model_data.get("display_name", model_name)
                
                # Parse capabilities and tags
                capabilities = model_data.get("capabilities", [])
                tags = model_data.get("tags", [])
                recommended_for = list(set(capabilities + tags))
                
                # Create ModelInfo object
                model_info = ModelInfo(
                    provider=provider,
                    model_name=model_name,
                    display_name=display_name,
                    description=model_data.get("description", f"{display_name} model"),
                    context_window=model_data.get("context_window", model_data.get("context_length", 0)),
                    recommended_for=recommended_for,
                    is_multimodal=model_data.get("multimodal", model_data.get("supports_vision", False)),
                    supports_tools=model_data.get("supports_tools", model_data.get("function_calling", True)),
                    cost_tier=model_data.get("pricing_tier", "standard"),
                    metadata=model_data
                )
                all_models.append(model_info)
            
            # Check if we got fewer results than requested (last page)
            if len(models_data) < per_page:
                break
            
            # Check if there's pagination info
            total = data.get("total", 0)
            if total > 0 and len(all_models) >= total:
                break
            
            page += 1
            
            # Safety limit to prevent infinite loops
            if page > 100:
                break
        
        self._models_cache = all_models
        return all_models
    
    async def fetch_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """Fetch models for a specific provider."""
        data = await self._make_request(f"providers/{provider}/models")
        models_data = data.get("models", [])
        
        models = []
        for model_data in models_data:
            model_name = model_data.get("name", model_data.get("id", ""))
            display_name = model_data.get("display_name", model_name)
            
            model_info = ModelInfo(
                provider=provider,
                model_name=model_name,
                display_name=display_name,
                description=model_data.get("description", f"{display_name} model"),
                context_window=model_data.get("context_window", 0),
                recommended_for=model_data.get("capabilities", []),
                is_multimodal=model_data.get("multimodal", False),
                supports_tools=model_data.get("supports_tools", True),
                cost_tier=model_data.get("pricing_tier", "standard"),
                metadata=model_data
            )
            models.append(model_info)
        
        return models
    
    def clear_cache(self):
        """Clear the cached models and providers."""
        self._models_cache = None
        self._providers_cache = None


# Fallback models
FALLBACK_MODELS: List[ModelInfo] = [
    ModelInfo(
        provider="openai",
        model_name="gpt-4o",
        display_name="GPT-4o",
        description="Most capable OpenAI model",
        context_window=128000,
        recommended_for=["chat", "code", "reasoning"],
        is_multimodal=True,
    ),
    ModelInfo(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        description="Best balance of intelligence and speed",
        context_window=200000,
        recommended_for=["chat", "code", "reasoning"],
        is_multimodal=True,
    ),
    ModelInfo(
        provider="google",
        model_name="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        description="Latest Gemini with fast inference",
        context_window=1000000,
        recommended_for=["chat", "code"],
        is_multimodal=True,
    ),
]


# Global registry instance (will be initialized with API key)
_global_registry: Optional[BackboardModelRegistry] = None


def initialize_registry(api_key: str):
    """Initialize the global model registry with an API key."""
    global _global_registry
    _global_registry = BackboardModelRegistry(api_key)
    return _global_registry


def get_registry() -> Optional[BackboardModelRegistry]:
    """Get the global registry instance."""
    return _global_registry


# Helper functions for compatibility
def search_models(query: str, models: List[ModelInfo], limit: int = 10) -> List[ModelInfo]:
    """
    Search models by query string.
    Matches against provider, model name, display name, description, and tags.
    """
    if not query:
        return models[:limit]
    
    query_lower = query.lower().strip()
    
    # Score each model based on match quality
    scored = []
    for model in models:
        score = 0
        search_text = model.search_text
        
        # Exact match in model name (highest priority)
        if query_lower in model.model_name.lower():
            score += 100
        
        # Match in provider name
        if query_lower in model.provider.lower():
            score += 50
        
        # Match in display name
        if query_lower in model.display_name.lower():
            score += 40
        
        # Match in description
        if query_lower in model.description.lower():
            score += 20
        
        # Match in recommended_for tags
        for tag in model.recommended_for:
            if query_lower in tag.lower():
                score += 30
        
        if score > 0:
            scored.append((score, model))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [m for _, m in scored[:limit]]


def get_all_providers(models: List[ModelInfo]) -> List[str]:
    """Get list of all available providers from models list."""
    providers = sorted(set(m.provider for m in models))
    return providers


def get_models_by_provider(provider: str, models: List[ModelInfo]) -> List[ModelInfo]:
    """Get all models for a specific provider."""
    return [m for m in models if m.provider == provider]


def get_model_by_id(provider: str, model_name: str, models: List[ModelInfo]) -> Optional[ModelInfo]:
    """Get a specific model by provider and name."""
    for model in models:
        if model.provider == provider and model.model_name == model_name:
            return model
    return None


def get_recommended_models(use_case: str, models: List[ModelInfo]) -> List[ModelInfo]:
    """Get models recommended for a specific use case."""
    use_case_lower = use_case.lower()
    return [m for m in models if use_case_lower in m.recommended_for]
