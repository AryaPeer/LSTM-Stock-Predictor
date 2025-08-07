import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


class ModelRepository:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        registry_file = self.model_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        registry_file = self.model_dir / "registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self._registry, f, indent=2, default=str)

    def save_model(self,
                   model: Any,
                   ticker: str,
                   model_type: str,
                   metadata: Optional[Dict] = None) -> str:
        # Generate unique model ID
        model_id = self._generate_model_id(ticker, model_type)

        # Save model file
        model_path = self.model_dir / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Update registry
        self._registry[model_id] = {
            'ticker': ticker,
            'model_type': model_type,
            'path': str(model_path),
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_registry()

        logger.info(f"Saved model {model_id}")
        return model_id

    def load_model(self, model_id: str) -> Optional[Any]:
        if model_id not in self._registry:
            logger.warning(f"Model {model_id} not found in registry")
            return None

        model_info = self._registry[model_id]
        model_path = Path(model_info['path'])

        if not model_path.exists():
            logger.error(f"Model file {model_path} not found")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Loaded model {model_id}")
        return model

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._registry.get(model_id)

    def list_models(self,
                    ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        models = []
        for model_id, info in self._registry.items():
            if ticker is None or info['ticker'] == ticker:
                models.append({
                    'id': model_id,
                    **info
                })
        return sorted(models, key=lambda x: x['created_at'], reverse=True)

    def delete_model(self, model_id: str) -> bool:
        if model_id not in self._registry:
            return False

        model_info = self._registry[model_id]
        model_path = Path(model_info['path'])

        # Delete file
        if model_path.exists():
            model_path.unlink()

        # Remove from registry
        del self._registry[model_id]
        self._save_registry()

        logger.info(f"Deleted model {model_id}")
        return True

    def _generate_model_id(self, ticker: str, model_type: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{ticker}_{model_type}_{timestamp}"
        model_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{ticker}_{model_type}_{timestamp}_{model_hash}"
