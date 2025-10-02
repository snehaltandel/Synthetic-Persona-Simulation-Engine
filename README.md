# Synthetic Persona Simulation Engine

The Synthetic Persona Simulation Engine is a prototype analytics toolkit that models user personas from behavioral data, simulates product changes, and produces executive-ready reports before features reach production.

## Project Objectives

- Build synthetic personas from clickstream and attribute datasets.
- Run "what-if" simulations to compare onboarding flows, UI changes, or feature releases.
- Estimate impact on drop-off, conversion, and engagement metrics.

## Repository Structure

```
├── app/                         # Streamlit demo app
├── data/                        # Sample datasets
├── notebooks/                   # Jupyter notebooks for experiments
├── reports/                     # Generated presentation artifacts
├── src/synthetic_persona_engine # Core Python package
│   ├── analytics/               # Reporting & visualization helpers
│   ├── data/                    # ETL + feature engineering
│   ├── modeling/                # Segmentation & trait inference
│   └── simulation/              # Persona generators and flow sims
├── pyproject.toml               # Package metadata for installation
└── requirements.txt             # Python dependencies
```

## Getting Started

1. **Install dependencies & package**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

   The editable install registers the `synthetic_persona_engine` package so that CLI tools and notebooks can import it.

2. **Run the end-to-end demo**

   ```bash
   python -m synthetic_persona_engine.cli --events data/sample_events.csv
   ```

   The command ingests the sample dataset, clusters personas, runs a simulated A/B test, and stores generated reports in the `reports/` directory.

3. **Explore interactively**

   Launch the Streamlit dashboard to upload your own dataset and explore persona simulations visually:

   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Experiment in notebooks**

   The `notebooks/persona_simulation_demo.ipynb` notebook walks through the workflow in a reproducible environment.

## Key Features

- **Data Layer**: ETL functions ingest CSV files, normalise schema, and engineer session, path, and user-level features.
- **Persona Modeling**: Encode event sequences, cluster personas, infer traits, and estimate their impact on conversion.
- **Simulation Engine**: Generate synthetic event sequences via a Markov chain, apply hypothetical product changes, and run simulated A/B tests.
- **Analytics & Reporting**: Build persona summaries, visualise funnels, generate Sankey diagrams, and export PowerPoint decks.

## Sample Workflow

```python
from synthetic_persona_engine.data import etl, features
from synthetic_persona_engine.modeling import segmentation
from synthetic_persona_engine.simulation import persona_generator, environment, experiments
from synthetic_persona_engine.analytics import reporting

# Load & engineer features
raw_events = etl.load_event_log_csv("data/sample_events.csv")
session_features = features.engineer_session_features(raw_events)
path_features = features.engineer_path_features(
    raw_events,
    funnel_events=["landing", "signup_start", "signup_complete"],
)
user_features = features.engineer_user_features(raw_events, session_features, path_features)

# Model personas
_, user_ids, _ = segmentation.encode_event_sequences(raw_events)
segmentation_result = segmentation.cluster_event_sequences(raw_events, n_clusters=2)
segmented_events = segmentation.attach_clusters(raw_events, segmentation_result, user_ids)
persona_summary = reporting.build_persona_summary(segmented_events)

# Simulate a product change
generator = persona_generator.MarkovPersonaGenerator.from_events(raw_events)
simulator = environment.ProductFlowSimulator(
    generator,
    completion_events={"signup_complete"},
    drop_events={"drop_off", "friction_drop"},
)
modification = environment.FlowModification(
    description="Add two extra onboarding clicks",
    additional_steps=2,
    friction_factor=0.1,
)
experiment = experiments.run_ab_experiment(
    simulator,
    n_personas=500,
    modification=modification,
    max_length=10,
    random_state=42,
)
report_df = reporting.experiment_summary(experiment)
```

## Stretch Goals

- Drift detection for monitoring persona distribution changes over time.
- Fairness diagnostics to ensure personas represent diverse groups.
- FastAPI endpoints to expose persona simulations to downstream systems.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
