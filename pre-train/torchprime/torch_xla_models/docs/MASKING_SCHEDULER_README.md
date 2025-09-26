# Masking Scheduler Documentation

The masking scheduler provides flexible control over masking probabilities and block sizes during training, supporting both constant and linear scheduling strategies for curriculum learning.

## Overview

The masking scheduler controls:
- **Three types of masking probabilities**:
  - **Prefix masking**: Randomly masks prefixes of sequences
  - **Truncate masking**: Randomly truncates sequences and fills with padding
  - **Block masking**: Masks contiguous blocks of tokens
- **Block size scheduling**: Progressively increases the available block sizes for more challenging masking patterns

## Configuration

### Constant Schedule

Use fixed masking probabilities throughout training:

```yaml
model:
  prefix_probability: 0.1
  truncate_probability: 0.1
  block_masking_probability: 0.2
  
  masking_scheduler:
    schedule_type: "constant"
    max_schedule_steps: null  # Not needed for constant schedule
```

### Linear Schedule

Gradually increase masking probabilities from 0 to target values:

```yaml
model:
  prefix_probability: 0.2         # Target probability
  truncate_probability: 0.15      # Target probability
  block_masking_probability: 0.3  # Target probability
  
  masking_scheduler:
    schedule_type: "linear"
    max_schedule_steps: 10000  # Steps to reach target probabilities
```

### Block Size Scheduling

You can schedule mask block sizes to progressively increase difficulty:

```yaml
model:
  # Scheduled block sizes - each sub-list is used for a portion of training
  mask_block_sizes: [[2], [2, 4], [2, 4, 8]]
  
  # Example: With 30,000 total training steps:
  # Steps 0-10,000: Only use block size 2
  # Steps 10,000-20,000: Use block sizes 2 or 4
  # Steps 20,000-30,000: Use block sizes 2, 4, or 8
```

Or use constant block sizes throughout training:

```yaml
model:
  # Constant block sizes - same sizes used throughout training
  mask_block_sizes: [2, 4, 8]
```

## How It Works

### Linear Schedule Behavior

1. **Steps 0 to max_schedule_steps**: Probabilities increase linearly from 0 to target values
2. **Steps beyond max_schedule_steps**: Probabilities remain at target values

Example progression for `prefix_probability: 0.2` with `max_schedule_steps: 1000`:
- Step 0: 0.0
- Step 250: 0.05 (25% of target)
- Step 500: 0.1 (50% of target)
- Step 1000: 0.2 (100% of target)
- Step 2000: 0.2 (remains at target)

### Block Size Schedule Behavior

When using scheduled block sizes (list of lists), the training is divided into equal phases:

Example with `mask_block_sizes: [[2], [2, 4], [2, 4, 8]]` and 30,000 total steps:
- Phase 1 (Steps 0-10,000): Only size 2 blocks can be masked
- Phase 2 (Steps 10,000-20,000): Sizes 2 or 4 blocks can be masked
- Phase 3 (Steps 20,000-30,000): Sizes 2, 4, or 8 blocks can be masked

This allows the model to:
1. Start with simple, small masked regions
2. Gradually learn to handle larger masked blocks
3. Eventually master complex masking patterns

### Integration with Training

The scheduler is automatically integrated into the training loop:

1. **Initialization**: Created when Trainer is instantiated
2. **Step Updates**: Automatically incremented after each training step
3. **Checkpointing**: State is saved/loaded with model checkpoints
4. **Model Forward**: Scheduled probabilities are passed to the model

## Usage Examples

### Using Pre-defined Configurations

```bash
# Use constant masking probabilities
python train.py model=flex-qwen-1b +model=masking/constant-schedule

# Use linear curriculum learning
python train.py model=flex-qwen-1b +model=masking/linear-schedule

# Use scheduled block sizes with linear probability ramp-up
python train.py model=flex-qwen-1b +model=masking/block-sizes-scheduled
```

### Custom Configuration

Create your own masking schedule:

```yaml
# my_custom_schedule.yaml
prefix_probability: 0.25
truncate_probability: 0.0    # Disable truncation
block_masking_probability: 0.4

masking_scheduler:
  schedule_type: "linear"
  max_schedule_steps: 20000  # Slower ramp-up
```

Then use:
```bash
python train.py model=flex-qwen-1b +model=masking/my_custom_schedule
```

## Checkpointing

The scheduler state is automatically saved and restored with model checkpoints:

```python
# Saved in checkpoint
{
    "model": model_state_dict,
    "optimizer": optimizer_state_dict,
    "scheduler": lr_scheduler_state_dict,
    "masking_scheduler": masking_scheduler_state_dict,  # Automatically included
    "step": current_step
}
```

When resuming from a checkpoint, the scheduler continues from the saved step with the correct probabilities.

## Implementation Details

### MaskingScheduler Class

```python
from torchprime.torch_xla_models.masking_scheduler import MaskingScheduler

# Create scheduler with block size scheduling
scheduler = MaskingScheduler(
    schedule_type="linear",
    max_schedule_steps=10000,
    prefix_probability=0.2,
    truncate_probability=0.15,
    block_masking_probability=0.3,
    mask_block_sizes=[[2], [2, 4], [2, 4, 8]],  # Scheduled block sizes
    total_training_steps=30000,
)

# Get current probabilities and block sizes
probs = scheduler.get_probabilities(step=15000)
# Returns: {
#     "prefix_probability": 0.2,          # At target (past max_schedule_steps)
#     "truncate_probability": 0.15,       # At target
#     "block_masking_probability": 0.3,   # At target
#     "mask_block_sizes": [2, 4]          # In phase 2 of block scheduling
# }

# Step forward
scheduler.step()

# Save/load state
state = scheduler.state_dict()
new_scheduler.load_state_dict(state)
```

### Model Integration

The model's forward method accepts `masking_config` parameter:

```python
def forward(
    self,
    input_ids,
    attention_mask=None,
    src_mask=None,
    training_mode="pretrain",
    masking_config=None,  # Scheduled probabilities
):
    # Use masking_config values if provided
    if masking_config is not None:
        prefix_prob = masking_config.get("prefix_probability", 0)
        truncate_prob = masking_config.get("truncate_probability", 0)
        block_prob = masking_config.get("block_masking_probability", 0)
    else:
        # Fall back to static config values
        prefix_prob = self.config.prefix_probability
        ...
```

## Benefits of Curriculum Learning

Linear scheduling enables curriculum learning where the model:
1. **Starts easy**: Initially trains on unmasked or lightly masked sequences
2. **Gradually increases difficulty**: Progressively introduces more masking
3. **Stabilizes training**: Reduces likelihood of early training instabilities
4. **Improves generalization**: Helps model learn robust representations

## Monitoring

Track masking probabilities in your logs:
- The scheduler logs initialization details
- Current probabilities can be logged during training
- WandB integration automatically tracks scheduler state

## Tips

1. **Choosing max_schedule_steps**: 
   - Typically 10-20% of total training steps
   - Shorter for fine-tuning, longer for pre-training

2. **Probability values**:
   - Start conservative (0.1-0.3 range)
   - Higher values = more aggressive masking
   - Balance between all three masking types

3. **Debugging**:
   - Set very small max_schedule_steps to quickly verify ramp-up
   - Use constant schedule to isolate masking effects
   - Monitor loss curves during ramp-up phase