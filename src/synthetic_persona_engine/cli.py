"""Command line interface for running the simulation pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from .analytics import reporting
from .data import etl, features
from .modeling import segmentation
from .simulation import environment, experiments, persona_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic Persona Simulation Engine")
    parser.add_argument("--events", type=Path, required=True, help="Path to the event log CSV file")
    parser.add_argument(
        "--funnel",
        nargs="*",
        default=["landing", "signup_start", "signup_complete"],
        help="Ordered funnel steps for completion analysis",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of persona clusters to generate",
    )
    parser.add_argument(
        "--personas",
        type=int,
        default=500,
        help="Number of personas to simulate per variant",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        help="Directory where reports should be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    events = etl.load_event_log_csv(args.events)
    session_features = features.engineer_session_features(events)
    path_features = features.engineer_path_features(events, funnel_events=args.funnel)
    features.engineer_user_features(events, session_features, path_features)

    _, user_ids, _ = segmentation.encode_event_sequences(events)
    segmentation_result = segmentation.cluster_event_sequences(events, n_clusters=args.clusters)
    segmented_events = segmentation.attach_clusters(events, segmentation_result, user_ids)
    persona_summary_df = reporting.build_persona_summary(segmented_events)

    generator = persona_generator.MarkovPersonaGenerator.from_events(events)
    simulator = environment.ProductFlowSimulator(
        generator,
        completion_events={args.funnel[-1]} if args.funnel else set(),
        drop_events={"drop_off", "friction_drop"},
    )
    modification = environment.FlowModification(
        description="Increase onboarding friction",
        additional_steps=2,
        friction_factor=0.08,
    )
    experiment_result = experiments.run_ab_experiment(
        simulator,
        n_personas=args.personas,
        modification=modification,
        max_length=10,
        random_state=42,
    )
    experiment_df = reporting.experiment_summary(experiment_result)

    persona_summary_path = args.output / "persona_summary.csv"
    experiment_path = args.output / "experiment_results.csv"
    pptx_path = args.output / "simulation_report.pptx"

    persona_summary_df.to_csv(persona_summary_path, index=False)
    experiment_df.to_csv(experiment_path, index=False)

    reporting.export_powerpoint(
        persona_summary=persona_summary_df,
        experiment_metrics=experiment_df,
        output_path=pptx_path,
    )

    print(f"Persona summary saved to {persona_summary_path}")
    print(f"Experiment results saved to {experiment_path}")
    print(f"PowerPoint report saved to {pptx_path}")


if __name__ == "__main__":
    main()
