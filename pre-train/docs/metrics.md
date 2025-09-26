# How is training performance measured

Training performance is a multi-faceted topic. `torchprime` measures the
following metrics, and saves them at `${output_dir}/train_metrics.json` at the
end of a training run.

### `train_runtime`

The total time of the training run (including compilation). In the JSON metrics,
it is a floating point representing the number of seconds of this training run.

### `step_execution_time`

The average time to execute a training step.

`torchprime` relies on the profiler to obtain the most accurate accelerator
timing. At the end of a training run, if a profile was taken, the trainer will
compute the time delta between the start of every training step on the
accelerator, and the median value of those time delta will be used as the
`step_execution_time`.

To obtain the step execution time, set a non-negative value in the `profile_step`
CLI argument to the trainer. Typically, you should use `profile_step` of at least
2 to skip over the first two compilation steps, which introduces large idle gaps
on the accelerator.

If the training is lazy tensor tracing bound or data loader bound,
`step_execution_time` will be larger than the time it takes the accelerator to do
the math.

In the JSON metrics, it is a floating point representing the number of seconds.
