# Local Measurement Experiments

This folder contains the experiments used to evaluate local region-measurement methods and their scaling behavior.

The experiment scripts generate CSV files with probability estimates and circuit metrics. The processing scripts convert these results into plots.

## Running the experiments

Run the scripts from this directory using `uv`.

### Classical Sampling and OBPA

`scaling_sampling.py` evaluates shot scaling for classical sampling and OBPA.

Available methods:

- `SimpleSampling`
- `OBPA`

Available regions:

- `point`
- `zero`
- `interval`
- `cloud`

For example:

```bash
uv run scaling_sampling.py --method SimpleSampling --oracle cloud --runs 10
uv run scaling_sampling.py --method OBPA --oracle interval --runs 10
```

The resulting CSV files are written to:

```text
scaling_output/shot_scaling_<method>_<oracle>.csv
```

### IQAE

`scaling_iqae.py` evaluates IQAE scaling for different accuracy (`epsilon`) and failure-probability (`alpha`) settings.

Available regions:

- `point`
- `zero`
- `interval`
- `cloud`

For example:

```bash
uv run scaling_iqae.py --oracle cloud
```

Results are written to:

```text
scaling_output/scaling_iqae_<oracle>.csv
```

### QAE

`scaling_qae.py` evaluates canonical QAE for different oracle regions.

```bash
uv run scaling_qae.py --oracle point --runs 10
```

Results are written to:

```text
scaling_output/scaling_qae_<oracle>.csv
```

## Processing the results

After running the experiments, execute the main result-processing workflow with:

```bash
bash process_shot_results.sh
```

To generate logarithmic versions of the scaling plots, use:

```bash
bash process_shot_results.sh --log-scale
```

The generated PDF figures are stored in:

```text
scaling_plots/
```

Individual processing scripts can also be run directly. For example:

```bash
uv run process_results_iqae_scaling.py \
    scaling_output/scaling_iqae_point.csv point
```

```bash
uv run process_results_shot_scaling.py \
    scaling_output/shot_scaling_obpa_point.csv \
    obpa_point \
    --ground_true_value 0.111111
```
