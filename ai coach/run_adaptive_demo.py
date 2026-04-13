import argparse

from workout_executor import run_adaptive_workout


SAMPLE_PLAN = {
    "rounds": 2,
    "rest_between_rounds": 15,
    "exercises": [
        {
            "exercise": "push_up",
            "avg_set_time": 10,
            "total_sets": 3,
            "rest_seconds": 10,
        },
        {
            "exercise": "bodyweight_squat",
            "avg_set_time": 10,
            "total_sets": 3,
            "rest_seconds": 10,
        },
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an adaptive workout with live CLI feedback.")
    parser.add_argument(
        "--log",
        default="data/workout_session_log.json",
        help="Path to JSON session log output.",
    )
    args = parser.parse_args()

    run_adaptive_workout(SAMPLE_PLAN, log_path=args.log)
