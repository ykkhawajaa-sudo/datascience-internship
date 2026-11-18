# Architecture

```mermaid
flowchart TD
 A[M5 CSVs or synthetic demo] --> B[Data loader]
 B --> C[Calendar, price and lag features]
 C --> D[Naive baseline and LightGBM]
 D --> E[WAPE and RMSSE evaluation]
 E --> F[Residual prediction intervals]
 F --> G[Bottom-up hierarchy aggregation]
 G --> H[Store, state and national views]
 G --> I[OR-Tools inventory optimization]
 H --> J[Plotly Dash]
 I --> J
```

Historical item-store sales feed the feature and model pipeline. Validation forecasts are presented in the prototype dashboard. A production extension should generate recursive future covariates, save forecast runs, and include replenishment lead times.
