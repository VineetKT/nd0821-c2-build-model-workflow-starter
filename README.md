# Build an ML Pipeline for Short-Term Rental Prices in NYC

## Overview

You are working for a property management company renting rooms and properties for short periods of
time on various rental platforms. You need to estimate the typical price for a given property based
on the price of similar properties. Your company receives new data in bulk every week. The model needs
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project, such a pipeline is built.

### [W&B project link](https://wandb.ai/vineetkt/nyc_airbnb)

<br />

### GitHub Project repo [Release v1.0.3](https://github.com/VineetKT/nd0821-c2-build-model-workflow-starter/tree/1.0.3)

<br />

## Train the model on a new data sample

Let's now test that we can run the release using `mlflow` without any other pre-requisite. We will
train the model on a new sample of data that our company received (`sample2.csv`):

(be ready for a surprise, keep reading even if the command fails)

```bash
> mlflow run https://github.com/VineetKT/nd0821-c2-build-model-workflow-starter.git \
             -v 1.0.3 \
             -P hydra_options="etl.sample='sample2.csv'"
```

## License

[License](LICENSE.txt)
