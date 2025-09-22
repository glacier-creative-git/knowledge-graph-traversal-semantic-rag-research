#!/usr/bin/env python3
"""
Model Configuration and Management System
========================================

Centralized model configuration, instantiation, and validation for deepeval integration.
Supports multiple LLM providers with proper error handling and caching.

Key Features:
- Multi-provider support: OpenAI, Ollama, Anthropic, OpenRouter
- Model validation and availability checking
- Intelligent caching to avoid re-instantiation
- Environment variable management
- Provider-specific configuration handling
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# DeepEval imports
from deepeval.models.base_model import DeepEvalBaseModel, DeepEvalBaseLLM
from deepeval.models import GPTModel, OllamaModel, AnthropicModel
import openai
import json
from pydantic import BaseModel

# Environment variable management
from dotenv import load_dotenv

# Data validation
from pydantic import ValidationError


class OpenRouterModel(DeepEvalBaseLLM):
    """
    Custom DeepEval model for OpenRouter integration.

    Implements the DeepEvalBaseLLM interface to properly work with DeepEval's
    evaluation framework while using OpenRouter's unified API.
    """

    def __init__(self, model_name: str, api_key: str, temperature: float = 0.1, max_tokens: int = 2000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1/",
            api_key=api_key
        )

    def load_model(self):
        """Return the OpenAI client for OpenRouter."""
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None) -> str:
        """
        Generate text using OpenRouter model.

        Args:
            prompt: The input prompt
            schema: Optional Pydantic schema for structured output

        Returns:
            Generated text response
        """
        client = self.load_model()

        messages = [{"role": "user", "content": prompt}]

        # Prepare request kwargs
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        # Add structured output if schema provided
        if schema:
            request_kwargs["response_format"] = {"type": "json_object"}
            # Also add a system message to encourage proper JSON formatting
            request_kwargs["messages"] = [
                {"role": "system", "content": "Always respond with valid JSON only. Do not include any text outside the JSON structure."},
                {"role": "user", "content": prompt}
            ]

        try:
            response = client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content

            # If schema provided, validate and return structured output
            if schema:
                try:
                    # First try to parse as JSON directly
                    return schema.model_validate_json(content)
                except:
                    try:
                        # For DeepEval's Response schema, handle specially
                        if hasattr(schema, 'model_fields') and 'response' in schema.model_fields:
                            # If JSON parsing fails, clean the content before wrapping
                            clean_content = self._clean_response_content(content)
                            return schema(response=clean_content)
                        else:
                            # For other schemas, return raw content without aggressive cleaning
                            return content
                    except Exception as e:
                        # Final fallback: return raw content
                        return content

            return content

        except Exception as e:
            raise RuntimeError(f"OpenRouter API call failed: {e}")

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        """
        Async version of generate. For now, falls back to sync.

        Args:
            prompt: The input prompt
            schema: Optional Pydantic schema for structured output

        Returns:
            Generated text response or structured output
        """
        # For now, use sync version. Could be improved with async client
        return self.generate(prompt, schema)

    def get_model_name(self):
        """Return descriptive model name for logging."""
        return f"OpenRouter: {self.model_name}"

    def _clean_response_content(self, content: str) -> str:
        """
        Clean malformed response content before wrapping in Response schema.

        Removes excessive whitespace, repeated newlines, and extracts actual content.
        """
        import re

        # Remove excessive newlines and whitespace
        cleaned = re.sub(r'\n\s*\n\s*', '\n\n', content)  # Collapse multiple newlines
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces
        cleaned = cleaned.strip()

        # If content looks like it has JSON structure, try to extract the actual content
        if '{' in cleaned and '}' in cleaned:
            # Try to extract JSON content between first { and last }
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start != -1 and end > start:
                potential_json = cleaned[start:end]
                try:
                    import json
                    parsed = json.loads(potential_json)

                    # For DeepEval evolution responses, extract the actual content
                    if isinstance(parsed, dict):
                        # Common evolution patterns
                        if "rewritten_input" in parsed:
                            return parsed["rewritten_input"]
                        elif "input" in parsed:
                            return parsed["input"]
                        elif "question" in parsed:
                            return parsed["question"]
                        elif "expected_output" in parsed:
                            return parsed["expected_output"]
                        elif len(parsed) == 1:
                            # Single key-value pair, return the value
                            value = list(parsed.values())[0]
                            # Don't return empty values
                            if value and str(value).strip() != "{}":
                                return value

                    # If we can't extract meaningful content but JSON is valid, return as-is
                    if potential_json.strip() != "{}":
                        return potential_json
                except:
                    pass  # Not valid JSON, continue with cleaning

        # If we end up with just "{}", return the original uncleaned content
        if cleaned.strip() == "{}":
            return content

        # Limit length to prevent excessively long responses
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."

        return cleaned


@dataclass
class ModelConfig:
    """Configuration container for a specific model provider."""
    provider: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 200
    base_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_providers = ["openai", "ollama", "anthropic", "openrouter"]
        if self.provider.lower() not in valid_providers:
            raise ValueError(f"Unsupported provider: {self.provider}. Must be one of {valid_providers}")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"Invalid temperature: {self.temperature}. Must be between 0 and 2")
        
        if self.max_tokens < 1:
            raise ValueError(f"Invalid max_tokens: {self.max_tokens}. Must be positive")



class ModelManager:
    """
    Centralized model configuration, instantiation, and validation system.
    
    Handles multiple LLM providers with intelligent caching and comprehensive
    error handling. Integrates with environment variables and configuration files.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize ModelManager with configuration and optional logger.
        
        Args:
            config: Complete system configuration dictionary
            logger: Optional logger instance (creates default if None)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.deepeval_config = config.get('deepeval', {})
        
        # Load environment variables
        load_dotenv()
        
        # Model instance cache to avoid re-instantiation
        self._model_cache: Dict[str, DeepEvalBaseModel] = {}
        
        # Validate configuration structure
        self._validate_configuration()
        
        self.logger.info("ModelManager initialized successfully")
    
    def get_question_generation_model(self) -> DeepEvalBaseModel:
        """
        Get configured model for synthetic question generation.
        
        Returns:
            DeepEvalBaseModel: Ready-to-use model instance for question generation
            
        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If model instantiation fails
        """
        return self._get_model('question_generation')
    
    def get_evaluation_judge_model(self) -> DeepEvalBaseModel:
        """
        Get configured model for evaluation judging (LLM-as-a-judge).
        
        Returns:
            DeepEvalBaseModel: Ready-to-use model instance for evaluation
            
        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If model instantiation fails
        """
        return self._get_model('evaluation_judge')
    
    def validate_model_availability(self) -> Dict[str, bool]:
        """
        Validate that all configured models are accessible and functional.
        
        Returns:
            Dict[str, bool]: Mapping of model types to availability status
        """
        results = {}
        
        for model_type in ['question_generation', 'evaluation_judge']:
            try:
                model = self._get_model(model_type)
                
                # Test with simple prompt to verify functionality
                test_response = model.generate("Test prompt: respond with 'OK'")
                
                # Validate response is not empty
                if test_response and len(test_response.strip()) > 0:
                    results[model_type] = True
                    self.logger.info(f"✅ {model_type} model validation successful")
                else:
                    results[model_type] = False
                    self.logger.error(f"❌ {model_type} model returned empty response")
                    
            except Exception as e:
                results[model_type] = False
                self.logger.error(f"❌ {model_type} model validation failed: {e}")
        
        return results
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a configured model.
        
        Args:
            model_type: Type of model ('question_generation' or 'evaluation_judge')
            
        Returns:
            Dict containing model configuration and status information
        """
        if model_type not in ['question_generation', 'evaluation_judge']:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        model_config = ModelConfig(**self.deepeval_config['models'][model_type])
        
        return {
            'provider': model_config.provider,
            'model_name': model_config.model_name,
            'temperature': model_config.temperature,
            'max_tokens': model_config.max_tokens,
            'base_url': model_config.base_url,
            'cached': model_type in self._model_cache,
            'environment_ready': self._check_environment_for_provider(model_config.provider)
        }
    
    def _get_model(self, model_type: str) -> DeepEvalBaseModel:
        """
        Internal method to instantiate and cache models.
        
        Args:
            model_type: Type of model to retrieve
            
        Returns:
            DeepEvalBaseModel: Instantiated model ready for use
        """
        # Return cached instance if available
        if model_type in self._model_cache:
            self.logger.debug(f"Returning cached {model_type} model")
            return self._model_cache[model_type]
        
        # Create new model instance
        model_config = ModelConfig(**self.deepeval_config['models'][model_type])
        
        self.logger.info(f"Instantiating {model_type} model: {model_config.provider}/{model_config.model_name}")
        
        # Create provider-specific model
        if model_config.provider.lower() == "openai":
            model = self._create_openai_model(model_config)
        elif model_config.provider.lower() == "ollama":
            model = self._create_ollama_model(model_config)
        elif model_config.provider.lower() == "anthropic":
            model = self._create_anthropic_model(model_config)
        elif model_config.provider.lower() == "openrouter":
            model = self._create_openrouter_model(model_config)
        else:
            raise ValueError(f"Unsupported model provider: {model_config.provider}")
        
        # Cache and return
        self._model_cache[model_type] = model
        self.logger.info(f"✅ {model_type} model instantiated and cached successfully")
        
        return model
    
    def _create_openai_model(self, config: ModelConfig) -> GPTModel:
        """Create OpenAI GPT model instance with proper configuration."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )

        return GPTModel(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    def _create_ollama_model(self, config: ModelConfig) -> OllamaModel:
        """Create Ollama model instance with proper configuration."""
        base_url = config.base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        # Validate Ollama server is accessible
        self._validate_ollama_server(base_url)

        return OllamaModel(
            model=config.model_name,
            base_url=base_url,
            temperature=config.temperature,
            generation_kwargs={'temperature': config.temperature}
        )
    
    def _create_anthropic_model(self, config: ModelConfig) -> AnthropicModel:
        """Create Anthropic model instance with proper configuration."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )

        return AnthropicModel(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    def _create_openrouter_model(self, config: ModelConfig) -> OpenRouterModel:
        """Create OpenRouter model instance using custom DeepEvalBaseLLM implementation."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )

        # Use custom OpenRouterModel that properly implements DeepEvalBaseLLM
        return OpenRouterModel(
            model_name=config.model_name,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    def _validate_configuration(self) -> None:
        """Validate deepeval configuration structure."""
        if 'models' not in self.deepeval_config:
            raise ValueError("Missing 'models' section in deepeval configuration")
        
        required_models = ['question_generation', 'evaluation_judge']
        for model_type in required_models:
            if model_type not in self.deepeval_config['models']:
                raise ValueError(f"Missing {model_type} configuration in deepeval.models")
    
    def _check_environment_for_provider(self, provider: str) -> bool:
        """Check if environment is properly configured for a provider."""
        if provider.lower() == "openai":
            return bool(os.getenv('OPENAI_API_KEY'))
        elif provider.lower() == "ollama":
            return bool(os.getenv('OLLAMA_BASE_URL')) or True  # Default URL available
        elif provider.lower() == "anthropic":
            return bool(os.getenv('ANTHROPIC_API_KEY'))
        elif provider.lower() == "openrouter":
            return bool(os.getenv('OPENROUTER_API_KEY'))
        else:
            return False
    
    def _validate_ollama_server(self, base_url: str) -> None:
        """Validate that Ollama server is accessible."""
        try:
            import requests
            
            # Test connection to Ollama server
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama server not accessible at {base_url}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to connect to Ollama server at {base_url}. "
                f"Please ensure Ollama is running. Error: {e}"
            )
        except ImportError:
            self.logger.warning("requests package not available - skipping Ollama server validation")
    
    def clear_cache(self) -> None:
        """Clear model cache - useful for testing or configuration changes."""
        self._model_cache.clear()
        self.logger.info("Model cache cleared")
    
    def get_cache_status(self) -> Dict[str, bool]:
        """Get current cache status for all model types."""
        return {
            'question_generation': 'question_generation' in self._model_cache,
            'evaluation_judge': 'evaluation_judge' in self._model_cache
        }
