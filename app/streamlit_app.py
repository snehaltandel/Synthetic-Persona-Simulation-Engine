"""Streamlit dashboard for the Synthetic Persona Simulation Engine."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from synthetic_persona_engine.analytics import reporting, visualization
from synthetic_persona_engine.data import etl, features
from synthetic_persona_engine.modeling import segmentation
from synthetic_persona_engine.simulation import environment, experiments, persona_generator

st.set_page_config(page_title="Synthetic Persona Simulation Engine", layout="wide")
st.title("🧩 Synthetic Persona Simulation Engine")
st.write(
    "Upload clickstream data, generate synthetic personas, and test how onboarding changes impact conversion before shipping."
)


@st.cache_data
def load_events(file: io.BytesIO) -> pd.DataFrame:
    return etl.load_event_log_csv(file)


uploaded_file = st.file_uploader("Upload clickstream CSV", type=["csv"])
if uploaded_file is not None:
    events = load_events(uploaded_file)
else:
    st.info("Using bundled sample dataset.")
    events = etl.load_event_log_csv(Path("data/sample_events.csv"))

funnel_steps = st.multiselect(
    "Funnel steps",
    options=sorted(events["event_type"].unique()),
    default=["landing", "signup_start", "signup_complete"],
)
num_clusters = st.slider("Number of persona clusters", 2, 6, 3)
num_personas = st.slider("Personas per variant", 100, 2000, 500, step=100)
additional_steps = st.slider("Additional onboarding steps", 0, 5, 2)
friction_factor = st.slider("Friction factor", 0.0, 0.3, 0.08, step=0.01)

session_features = features.engineer_session_features(events)
path_features = features.engineer_path_features(events, funnel_events=funnel_steps or ["landing"])
user_features = features.engineer_user_features(events, session_features, path_features)

_, user_ids, _ = segmentation.encode_event_sequences(events)
segmentation_result = segmentation.cluster_event_sequences(events, n_clusters=num_clusters)
segmented_events = segmentation.attach_clusters(events, segmentation_result, user_ids)
persona_summary_df = reporting.build_persona_summary(segmented_events)

st.subheader("Persona Archetypes")
st.dataframe(persona_summary_df)

with st.expander("User-level behavioral features"):
    st.dataframe(user_features)

persona_generator_model = persona_generator.MarkovPersonaGenerator.from_events(events)
simulator = environment.ProductFlowSimulator(
    persona_generator_model,
    completion_events={funnel_steps[-1]} if funnel_steps else set(),
    drop_events={"drop_off", "friction_drop"},
)
modification = environment.FlowModification(
    description="Simulated onboarding change",
    additional_steps=additional_steps,
    friction_factor=friction_factor,
)
experiment_result = experiments.run_ab_experiment(
    simulator,
    n_personas=num_personas,
    modification=modification,
    max_length=10,
    random_state=42,
)
experiment_df = experiments.results_to_dataframe(experiment_result)

st.subheader("Simulation Outcomes")
st.dataframe(experiment_df)

col1, col2 = st.columns(2)
with col1:
    funnel_fig = visualization.funnel_chart(experiment_df[experiment_df["variant"] != "uplift"])
    st.plotly_chart(funnel_fig, use_container_width=True)
with col2:
    sankey_fig = visualization.sankey_from_sequences(experiment_result.treatment_sequences)
    st.plotly_chart(sankey_fig, use_container_width=True)

if st.button("Generate PowerPoint Report"):
    output_path = Path("reports/streamlit_report.pptx")
    output_path.parent.mkdir(exist_ok=True)
    reporting.export_powerpoint(
        persona_summary=persona_summary_df,
        experiment_metrics=experiment_df,
        output_path=output_path,
    )
    st.success(f"Report saved to {output_path}")
