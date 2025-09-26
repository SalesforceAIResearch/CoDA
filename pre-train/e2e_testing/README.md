# E2E testing

These scripts are used during the [E2E test][e2e-test] GitHub action to run some
models and validate the results.

## E2E test design

The workflows in [e2e_test.yml][e2e-test] does a few things:

- Set up `gcloud` credentials from a Service Account key managed in repo secrets.
- Install `torchprime`.
- Test `tp use` and point it to an XPK cluster hosted internally.
- Test `tp run` on a few models.

After kicking off the training of some models, it starts a parallel job for each
model, and runs a few checks. This is implemented in
[reusable_e2e_check.yml][e2e-check]:

- Stream the logs.
- Check workload exit code.
- Check for specific log strings that indicate training success.
- Check that there is a profile `.pb` file.
- Check that the step time is within a reasonable range.

## What to do when step time is out of range

As of https://github.com/AI-Hypercomputer/torchprime/pull/208, we'll start
checking that the step time of model training E2E tests falls within
experimentally derived bounds.

### If the step time falls below the lower bound

If this is the result of an optimization/updated deps, you may re-center the
bounds to the latest step time, keeping the confidence intervals unchanged.

If there were no code changes and the step time is still below the lower bound,
consider growing the confidence interval.

### If the step time goes above the upper bound

If this is the result of a code change, you may have introduced a regression.
Investigate the root cause and avoid the slow down before landing the PR.

If there were no code changes and the step time is still above the upper bound,
we'll need to discuss with the hardware teams because it may be the result of
hardware changes.

### Formula for computing the lower and upper bounds

Let $X = \{x_1, x_2, \ldots, x_n\}$ be the observed step time for a workload.
Let $n$ be the number of observations (e.g. training runs).
Let $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$ denote the sample mean.
Let $\min(X)$ be the minimum value in the dataset.
Let $\max(X)$ be the maximum value in the dataset.

Let $C_L$ be the desired confidence level, $0.999$.
Let $\alpha = 1 - C_L$ be the significance level.
Let $t_{\alpha/2, n-1}$ denote the critical value from the Student's
t-distribution for $n-1$ degrees of freedom and a two-tailed test.

The half-width of the confidence interval, $H$, is calculated as:

$$H = \max\left(t_{\alpha/2, n-1}, \quad 0.015 \cdot \bar{x}, \quad \max(X) - \bar{x}, \quad \bar{x} - \min(X) \right)$$

We estimate the bounds via Student's t-distribution, clamped to be not smaller
than 1.5% of the step time and not rule out any past results, to avoid flakes.

The final lower and upper bounds of the step time are:

$$\text{Interval} = [ \bar{x} - H, \quad \bar{x} + H ]$$

## v6e XPK cluster

E2E tests are launched onto an XPK cluster named `tpu-v6e-ci`.

(Googlers only) To heal or re-create the cluster, use the following:

```sh
xpk cluster create \
    --tpu-type v6e-4 \
    --cluster tpu-v6e-ci \
    --num-slices 64 \
    --on-demand \
    --zone us-central2-b \
    --project tpu-pytorch \
    --default-pool-cpu-machine-type=n2-standard-32
```

### Secret management

In order for GitHub to launch workloads in the `tpu-v6e-ci` XPK cluster, the
workflow uses a `${{ secrets.GCP_SA_KEY }}` service account key. This key will
expire every 3 months and need to be manually rotated.

(Googlers only) When GitHub Actions fail with this error:

```
ERROR: (gcloud.auth.activate-service-account) There was a problem refreshing auth tokens for account [...]: ('invalid_grant: Invalid JWT Signature.', ***'error': 'invalid_grant', 'error_description': 'Invalid JWT Signature.'***)
```

Visit http://shortn/_uWi00zol5Q to mint a new key, and manually enter that into
the `GCP_SA_KEY` secrets field in the "Secrets and variables > Actions" field of
`torchprime` repository settings.

[e2e-test]: /.github/workflows/e2e_test.yml
[e2e-check]: /.github/workflows/reusable_e2e_check.yml
