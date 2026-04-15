"""
Utilities for working with Optuna studies.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import optuna


def save_study(
    study: optuna.Study,
    output_dir: str,
    save_plots: bool = True,
) -> None:
    """
    Save an Optuna study to disk.
    
    Args:
        study: Optuna study to save
        output_dir: Directory to save study files
        save_plots: Whether to generate and save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study object
    study_path = output_dir / "study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    # Save best params as JSON
    best_params_path = output_dir / "best_params.json"
    best_params = {
        "study_name": study.study_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    # Save all trials
    trials_path = output_dir / "trials.csv"
    df = study.trials_dataframe()
    df.to_csv(trials_path, index=False)
    
    print(f"Study saved to {output_dir}")
    print(f"  - Study object: {study_path}")
    print(f"  - Best params: {best_params_path}")
    print(f"  - Trials CSV: {trials_path}")
    
    # Generate plots
    if save_plots:
        plot_study_results(study, output_dir)


def load_study(study_path: str) -> optuna.Study:
    """
    Load an Optuna study from disk.
    
    Args:
        study_path: Path to saved study pickle file
    
    Returns:
        Loaded Optuna study
    """
    with open(study_path, "rb") as f:
        study = pickle.load(f)
    return study


def plot_study_results(
    study: optuna.Study,
    output_dir: str,
) -> None:
    """
    Generate and save plots for an Optuna study.
    
    Args:
        study: Optuna study
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_image(str(output_dir / "optimization_history.png"))
        
        # Parameter importance
        try:
            fig = vis.plot_param_importances(study)
            fig.write_image(str(output_dir / "param_importances.png"))
        except Exception as e:
            print(f"Could not plot parameter importances: {e}")
        
        # Parallel coordinate plot
        try:
            fig = vis.plot_parallel_coordinate(study)
            fig.write_image(str(output_dir / "parallel_coordinate.png"))
        except Exception as e:
            print(f"Could not plot parallel coordinate: {e}")
        
        # Slice plot
        try:
            fig = vis.plot_slice(study)
            fig.write_image(str(output_dir / "slice_plot.png"))
        except Exception as e:
            print(f"Could not plot slice: {e}")
        
        # Contour plot (for pairs of parameters)
        try:
            fig = vis.plot_contour(study)
            fig.write_image(str(output_dir / "contour_plot.png"))
        except Exception as e:
            print(f"Could not plot contour: {e}")
        
        print(f"Plots saved to {output_dir}")
        
    except ImportError:
        print("matplotlib or plotly not available, skipping plot generation")
    except Exception as e:
        print(f"Error generating plots: {e}")


def get_study_summary(study: optuna.Study) -> dict:
    """
    Get a summary of an Optuna study.
    
    Args:
        study: Optuna study
    
    Returns:
        Dictionary with study summary
    """
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
    }
    
    if study.best_trial:
        summary["best_trial"] = study.best_trial.number
        summary["best_value"] = study.best_value
        summary["best_params"] = study.best_params
    
    return summary


def print_study_summary(study: optuna.Study) -> None:
    """
    Print a summary of an Optuna study.
    
    Args:
        study: Optuna study
    """
    summary = get_study_summary(study)
    
    print("=" * 60)
    print("Study Summary")
    print("=" * 60)
    print(f"Name: {summary['study_name']}")
    print(f"Direction: {summary['direction']}")
    print(f"Total trials: {summary['n_trials']}")
    print(f"  Complete: {summary['n_complete']}")
    print(f"  Pruned: {summary['n_pruned']}")
    print(f"  Failed: {summary['n_failed']}")
    
    if "best_trial" in summary:
        print(f"\nBest trial: {summary['best_trial']}")
        print(f"Best value: {summary['best_value']:.4f}")
        print("\nBest parameters:")
        for key, value in summary['best_params'].items():
            print(f"  {key}: {value}")
    
    print("=" * 60)


def compare_studies(
    studies: dict,
    metric: str = "best_value",
) -> dict:
    """
    Compare multiple Optuna studies.
    
    Args:
        studies: Dictionary of study name to study object
        metric: Metric to compare ("best_value", "n_trials", etc.)
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    for name, study in studies.items():
        summary = get_study_summary(study)
        comparison[name] = summary
    
    # Find best study
    if metric == "best_value":
        # For minimize, lower is better
        best_study = min(comparison.items(), key=lambda x: x[1].get("best_value", float("inf")))
    else:
        best_study = None
    
    return {
        "comparison": comparison,
        "best_study": best_study[0] if best_study else None,
    }
