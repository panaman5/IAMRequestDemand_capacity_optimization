# IAM Workload Capacity Model

An independent modeling repository for the Project Optimus IAM capacity
problem. It currently contains the modeling contract and end-to-end flow only;
testing, data connectors, dashboards, and production hardening are deliberately
left for a later phase.

## Core idea

The system protects the capacity required by IAM requests first. Other objects
may use only the remaining capacity.

```text
request observations
        |
        v
per-request workload weights
        |
        +--> request-count model ------------------+
        |                                           |
        +--> approval outcome model ---------------+--> compound workload forecast
        |                                           |
        +--> conditional workload distributions ---+
                                                    |
                                                    v
                                  protected request capacity (P95)
                                                    |
                                                    v
                                      spare capacity or deficit
```

## Model contract

For request `i`:

```text
w_i = w_i_pre + I(approved) * w_i_backend + epsilon_i
```

Backend workload is measured as capacity-unit time:

```text
w_i_backend = sum_j(r_ij * tau_ij) + r_i_wf * T_i_wf
```

Parallel backend tasks are summed because they consume capacity concurrently.
Approval waiting time is not active capacity workload unless the waiting state
actually reserves a capacity unit.

For time bucket `t`:

```text
W_t = sum_i(w_i,t)
```

## Forecasting flow

1. Fit a Negative Binomial baseline to historical request counts.
2. Fit a binomial rejection-rate component from approval decisions.
3. Build recent-data-weighted empirical workload distributions for approved and
   rejected requests.
4. Forecast request count, approval outcome, and per-request weight.
5. Sum simulated request weights to generate the predictive distribution of
   total request workload.
6. Reserve a selected workload quantile, such as P95, for requests.
7. Calculate spare capacity as total available capacity minus protected request
   capacity.

The empirical distributions are used instead of assuming that request weights
are Normal. The Negative Binomial model handles request-count overdispersion;
the workload forecast is a compound sum.

## Run the modeling example

The example uses only the Python standard library:

```bash
python3 examples/run_flow.py
```

## Repository map

- `src/iam_workload_capacity_model/models.py` - request and backend-task data contracts.
- `src/iam_workload_capacity_model/workload.py` - request weights and empirical distributions.
- `src/iam_workload_capacity_model/forecast.py` - Negative Binomial, rejection, and compound forecast components.
- `src/iam_workload_capacity_model/capacity.py` - protected request capacity and spare-capacity calculation.
- `examples/run_flow.py` - minimal end-to-end modeling flow.
- `docs/flow.md` - conceptual sequence and the transition to the future queue model.
