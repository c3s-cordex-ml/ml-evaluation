# Evaluation of ML-RCMs

This module builds on the set of evaluation diagnostics recommended by CORDEX-ML-Bench for assessing ML-based regional climate downscaling models: 


<a href="https://github.com/WCRP-CORDEX/ml-benchmark/evaluation/" target="_blank">https://github.com/WCRP-CORDEX/ml-benchmark/evaluation/</a>

**Indices** quantify particular characteristics or properties of a single simulation `x`, for example, the mean, specific quantiles, or extremes. These indices are always calculated for an individual simulation and often form the foundation for various diagnostics.

**Diagnostics** compare two simulations, `x0` and `x1`, which correspond to the ground-truth (reference) data and the model predictions, respectively. These diagnostics are designed to evaluate different aspects of model performance. For example, to assess how well a model captures warm extremes, one might use the bias of the TXx index. Some diagnostics rely on specific indices, while others, such as RMSE, PSD, or the Wasserstein distance, do not depend on them.


## Indices
Initial list of metrics to validate RCM emulators. This framework is adapted from Kendon et al. (2025).


| Category | Index | Input Timescale | Variables | Output Timescale | Performance Diagnostic | Scoreboard Summary |
|-------|-------|-------|-------|-------|-------|-------|
| **Basic performance** | Raw output. | sub-daily, daily | All | as input | Mean Bias, RMSE, Corr over time (F) | Spatial mean? |
|  |  |  |  |  | ***QQ plots*** |  |
|   | Wet/dry day frequency | daily | precip | Freq per year or season | Mean Bias, RMSE, Corr over time (F) | Spatial mean? |
| **Extremes** | Percentiles (e.g. 95th, 99th) | daily, sub-daily | All | Annual/Seasonal, compute over whole dataset? | Bias of single fields (F) | Spatial mean? |
|   | Max (min) values | daily | All | Max per year or season | Mean Bias, RMSE, Corr over time (F) | Spatial mean? |
|   | Frequency of threshold exceedance | daily | All | Freq per year or season | Mean Bias, RMSE, Corr over time (F) | Spatial mean? |
|   |  |  |  |  | ***Fractional Skill Score*** |  |
| **Spatial coherence** | Radially averaged power | daily, sub-daily | All |  |  |  |
| **Temporal coherence** | Autocorrelation | daily | All | Annual/Seasonal, compute over whole dataset? | Bias of single fields (F) | Spatial mean? |
|   | Wet/dry, hot/cold spell length: max ***(median, 90th percentile?)*** | daily | precip, temp | Max per year or season | Mean Bias, RMSE, Corr over time (F) | Spatial mean? |
|   | Power spectral density | daily | All | Annual/Seasonal, compute over whole dataset? | PSD plots |  |
|   | ***Diurnal cycle (amplitude, phase)*** | sub-daily | precip, temp |  |  |  |
|   | ***Joint distribution characteristics e.g. mean, variance*** | sub-daily | All |  | ***Wasserstein Distance for 2d distributions (F?)*** |  |
| **Multivariate coherence** | Multivariate correlation | daily | e.g. temp-precip | Annual/Seasonal, compute over whole dataset? | Bias of single fields (F) | Spatial mean? |
|  | ***Probability of joint exceedances*** | daily | e.g. temp-precip | Annual/Seasonal, compute over whole dataset? | Bias of single fields (F) | Spatial mean? |
|  | ***Joint distribution*** | daily | e.g. temp-precip | Annual/Seasonal, compute over whole dataset? | ***Wasserstein Distance for 2d distributions (F?)*** |  |
| **Climate change** | ***Long-term trends e.g. of monthly means, extreme percentiles*** | daily | All | Annual/Seasonal, compute over whole dataset? | Bias of single fields (F) | Spatial mean? |
|  |  |  |  |  |  |  |
| **Case studies** | ***Event based (compound extremes; multi-day events)*** | daily | e.g. heat+drought |  |  |  |
| **Object-Oriented** | ***Raw output + object identifying algorithm*** | daily | precip, temp |  | ***SAL Score*** |  |
| **Explainability** | ***Integrated gradients*** | daily | All |  |  |  |


## Performance Diagnostics
Bias, RMSE, correlation (over time dimension; over spatial dimension?).

***Wasserstein Distance*** (for single or ***joint distributions***).

***Fractional skill score (for threshold exceedances)***.

***SAL score (structure, amplitude, location)***.

***Probabilistic Metrics? Ensemble spread; Continuous rank probability score (CRPS)***.


## Scoreboard summary statistics
Spatial mean? (where performance diagnostic outputs 1 field).

Bias, RMSE, correlation over spatial dimension? (where index outputs 1 field for prediction and 1 field for target/truth).


## References
Kendon, E. J., Addison, H., Doury, A., Somot, S., Watson, P. A., Booth, B. B., ... & Scullion, C. (2025). Potential for machine learning emulators to augment regional climate simulations in provision of local climate change information. Bulletin of the American Meteorological Society, BAMS-D.
