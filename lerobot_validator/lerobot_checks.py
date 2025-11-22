"""
Checks specific to lerobot dataset format.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from cloudpathlib import CloudPath, AnyPath
from lerobot_validator.schemas import REQUIRED_LEROBOT_FIELDS


class LerobotDatasetChecker:
    """Performs checks on the lerobot dataset itself."""

    def __init__(self, dataset_path: Union[str, Path, CloudPath]):
        """
        Initialize the lerobot dataset checker.

        Args:
            dataset_path: Path to the lerobot dataset directory (supports local or cloud paths)
        """
        if isinstance(dataset_path, str):
            self.dataset_path = AnyPath(dataset_path)
        else:
            self.dataset_path = dataset_path
        self.errors: List[str] = []
        self.episode_info: Dict[str, Any] = {}

    def validate(self) -> bool:
        """
        Validate the lerobot dataset.

        Returns:
            True if validation passes, False otherwise
        """
        self.errors = []

        # Check if dataset directory exists
        if not self.dataset_path.exists():
            self.errors.append(f"Dataset directory not found: {self.dataset_path}")
            return False

        # Check for required lerobot fields (task and frequency)
        self._check_required_lerobot_fields()

        # Load episode information (timestamps, duration)
        self._load_episode_info()

        return len(self.errors) == 0

    def _check_required_lerobot_fields(self) -> None:
        """Check that required fields (task, frequency) exist in the lerobot dataset."""
        # Look for info.json which typically contains dataset-level information
        info_file = self.dataset_path / "info.json"
        
        if not info_file.exists():
            self.errors.append(
                f"Required file not found: {info_file}. "
                "The lerobot dataset must contain an info.json file."
            )
            return
        
        try:
            with info_file.open("r") as f:
                info = json.load(f)
            
            # Check for task field
            if "task" not in info and "tasks" not in info:
                self.errors.append(
                    "Missing 'task' field in info.json. "
                    "The lerobot dataset must include a task string for every episode."
                )
            
            # Check for frequency field
            if "frequency" not in info and "fps" not in info:
                self.errors.append(
                    "Missing 'frequency' field in info.json. "
                    "The lerobot dataset must specify the data collection frequency."
                )
                
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in info.json: {e}")
        except Exception as e:
            self.errors.append(f"Failed to read info.json: {e}")

    def _load_episode_info(self) -> None:
        """
        Load episode information including timestamps and durations.
        This information will be used for cross-validation with annotations.
        """
        # Try to load from info.json
        info_file = self.dataset_path / "info.json"
        if info_file.exists():
            try:
                with info_file.open("r") as f:
                    info = json.load(f)
                    if "episodes" in info:
                        self.episode_info = info["episodes"]
            except Exception:
                pass

        # Try to load from meta directory
        meta_dir = self.dataset_path / "meta"
        if meta_dir.exists():
            episode_files = list(meta_dir.glob("episode_*.json"))
            for ep_file in episode_files:
                try:
                    with ep_file.open("r") as f:
                        ep_data = json.load(f)
                        episode_id = ep_data.get("episode_index") or ep_data.get("episode_id")
                        if episode_id is not None:
                            self.episode_info[str(episode_id)] = ep_data
                except Exception:
                    continue

    def get_episode_duration(self, episode_id: str) -> Optional[float]:
        """
        Get the duration of an episode in seconds.

        Args:
            episode_id: The episode identifier

        Returns:
            Duration in seconds, or None if not available
        """
        if episode_id in self.episode_info:
            ep = self.episode_info[episode_id]
            # Try different possible field names
            if "duration" in ep:
                return ep["duration"]
            if "length" in ep and "fps" in ep:
                return ep["length"] / ep["fps"]
            if "num_frames" in ep and "fps" in ep:
                return ep["num_frames"] / ep["fps"]
        return None

    def get_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.errors

