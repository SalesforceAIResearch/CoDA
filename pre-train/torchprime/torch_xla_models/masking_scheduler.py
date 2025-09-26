"""Masking probability scheduler for progressive curriculum learning."""

import logging
from typing import Dict, List, Optional, Union
import torch_xla.runtime as xr

logger = logging.getLogger(__name__)


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return xr.process_index() == 0


class MaskingScheduler:
    """Scheduler for masking probabilities that supports constant and linear schedules."""

    def __init__(
        self,
        schedule_type: str = "constant",
        max_schedule_steps: Optional[int] = None,
        prefix_probability: float = 0.0,
        truncate_probability: float = 0.0,
        block_masking_probability: float = 0.0,
        mask_block_sizes: Optional[Union[List[int], List[List[int]]]] = None,
        total_training_steps: Optional[int] = None,
    ):
        """
        Initialize the masking scheduler.

        Args:
            schedule_type: Either "constant" or "linear"
            max_schedule_steps: For linear schedule, number of steps to reach max probabilities
            prefix_probability: Target probability for prefix masking
            truncate_probability: Target probability for truncation
            block_masking_probability: Target probability for block masking
            mask_block_sizes: Either a list of ints (e.g., [2, 4, 8]) or a list of lists
                            for scheduled block sizes (e.g., [[2], [2, 4], [2, 4, 8]])
            total_training_steps: Total training steps, required when mask_block_sizes is a list of lists
        """
        self.schedule_type = schedule_type
        self.max_schedule_steps = max_schedule_steps

        # Target probabilities
        self.target_prefix_prob = prefix_probability
        self.target_truncate_prob = truncate_probability
        self.target_block_masking_prob = block_masking_probability

        # Current step counter
        self.current_step = 0

        # Handle mask block sizes scheduling
        self.mask_block_sizes = mask_block_sizes
        self.total_training_steps = total_training_steps
        self.scheduled_block_sizes = False
        self.block_size_boundaries = []

        if mask_block_sizes is not None and len(mask_block_sizes) > 0:
            # Check if it's a list of lists (scheduled)
            if isinstance(mask_block_sizes[0], list):
                self.scheduled_block_sizes = True
                if total_training_steps is None:
                    raise ValueError(
                        "total_training_steps is required when mask_block_sizes is a list of lists"
                    )
                # Calculate boundaries for each phase
                num_phases = len(mask_block_sizes)
                phase_length = total_training_steps / num_phases
                self.block_size_boundaries = [
                    int(phase_length * (i + 1)) for i in range(num_phases - 1)
                ]
                if is_main_process():
                    logger.info(
                        "Scheduled block sizes: %s with boundaries at steps: %s",
                        mask_block_sizes,
                        self.block_size_boundaries,
                    )
            else:
                if is_main_process():
                    logger.info("Using constant block sizes: %s", mask_block_sizes)

        # Validate configuration
        if schedule_type == "linear" and max_schedule_steps is None:
            raise ValueError(
                "Linear schedule requires max_schedule_steps to be specified"
            )

        if schedule_type not in ["constant", "linear"]:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        if is_main_process():
            logger.info(
                f"Initialized MaskingScheduler with {schedule_type} schedule. "
                f"Target probabilities - prefix: {prefix_probability}, "
                f"truncate: {truncate_probability},"
                f"block_masking: {block_masking_probability}, masking_block_sizes: {mask_block_sizes}"
            )
        if schedule_type == "linear" and is_main_process():
            logger.info(
                f"Linear schedule will reach targets in {max_schedule_steps} steps"
            )

    def get_current_block_sizes(self, step: int) -> Optional[List[int]]:
        """Get the current mask block sizes based on the step."""
        if self.mask_block_sizes is None:
            return None

        if not self.scheduled_block_sizes:
            # Simple list of block sizes, return as is
            return self.mask_block_sizes

        # Scheduled block sizes - determine which phase we're in
        phase = 0
        for i, boundary in enumerate(self.block_size_boundaries):
            if step < boundary:
                phase = i
                break
        else:
            # We're past all boundaries, use the last phase
            phase = len(self.mask_block_sizes) - 1

        return self.mask_block_sizes[phase]

    def get_schedule(
        self, step: Optional[int] = None
    ) -> Dict[str, Union[float, List[int]]]:
        """
        Get current masking probabilities and block sizes based on the schedule.

        Args:
            step: Optional step number. If None, uses internal step counter.

        Returns:
            Dictionary with current probabilities for each masking type and block sizes
        """
        if step is None:
            step = self.current_step

        result = {}

        if self.schedule_type == "constant":
            result.update(
                {
                    "prefix_probability": self.target_prefix_prob,
                    "truncate_probability": self.target_truncate_prob,
                    "block_masking_probability": self.target_block_masking_prob,
                }
            )

        elif self.schedule_type == "linear":
            # Linear interpolation from 0 to target over max_schedule_steps
            progress = min(1.0, step / self.max_schedule_steps)

            result.update(
                {
                    "prefix_probability": self.target_prefix_prob * progress,
                    "truncate_probability": self.target_truncate_prob * progress,
                    "block_masking_probability": self.target_block_masking_prob
                    * progress,
                }
            )

        # Add current block sizes
        current_block_sizes = self.get_current_block_sizes(step)
        if current_block_sizes is not None:
            result["mask_block_sizes"] = current_block_sizes

        return result

    def step(self):
        """Increment the internal step counter."""
        self.current_step += 1

    def state_dict(self) -> Dict:
        """Return state dict for checkpointing."""
        return {
            "schedule_type": self.schedule_type,
            "max_schedule_steps": self.max_schedule_steps,
            "target_prefix_prob": self.target_prefix_prob,
            "target_truncate_prob": self.target_truncate_prob,
            "target_block_masking_prob": self.target_block_masking_prob,
            "current_step": self.current_step,
            "mask_block_sizes": self.mask_block_sizes,
            "total_training_steps": self.total_training_steps,
            "scheduled_block_sizes": self.scheduled_block_sizes,
            "block_size_boundaries": self.block_size_boundaries,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint."""
        self.schedule_type = state_dict["schedule_type"]
        self.max_schedule_steps = state_dict["max_schedule_steps"]
        self.target_prefix_prob = state_dict["target_prefix_prob"]
        self.target_truncate_prob = state_dict["target_truncate_prob"]
        self.target_block_masking_prob = state_dict["target_block_masking_prob"]
        self.current_step = state_dict["current_step"]

        # Load mask block sizes configuration
        self.mask_block_sizes = state_dict.get("mask_block_sizes")
        self.total_training_steps = state_dict.get("total_training_steps")
        self.scheduled_block_sizes = state_dict.get("scheduled_block_sizes", False)
        self.block_size_boundaries = state_dict.get("block_size_boundaries", [])

        if is_main_process():
            logger.info("Loaded MaskingScheduler state at step %s", self.current_step)
