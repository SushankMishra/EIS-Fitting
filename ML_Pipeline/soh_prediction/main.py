import argparse
import os
from soh import config
from soh.data_processing import calculate_soh
from soh.feature_engineering import create_feature_table
from soh.model_training import train_model
from soh.inference import predict_soh

def main():
    """Main function to run the ML pipeline with model selection."""
    parser = argparse.ArgumentParser(description="SoH Prediction Pipeline")
    
    # Create subparsers for each step
    subparsers = parser.add_subparsers(dest="step", required=True, help="Pipeline step to run")

    # Parser for 'create_features'
    parser_features = subparsers.add_parser("create_features", help="Process raw data and generate features.")

    # Parser for 'train'
    parser_train = subparsers.add_parser("train", help="Train a model.")
    parser_train.add_argument(
        "--model", "-m",
        type=str,
        default='rf',
        choices=['rf', 'xgb', 'svm'],
        help="The model to train (rf, xgb, or svm)."
    )

    # Parser for 'predict'
    parser_predict = subparsers.add_parser("predict", help="Run inference with a trained model.")
    parser_predict.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=['rf', 'xgb', 'svm'],
        help="The trained model to use for prediction."
    )
    parser_predict.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to a single EIS file for prediction."
    )
    parser_predict.add_argument(
        "--soc",
        type=float,
        required=True,
        help="The State of Charge (e.g., 0.9 for 90%) for the EIS measurement."
    )

    args = parser.parse_args()

    # Ensure necessary directories exist
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(config.MODEL_PATH, exist_ok=True)

    if args.step == "create_features":
        print("--- Starting: SoH Calculation and Feature Engineering ---")
        soh_df = calculate_soh(config.DISCHARGE_DATA_PATH)
        create_feature_table(config.EIS_DATA_PATH, soh_df)
        print(f"--- Finished: Features saved to {config.FEATURE_PATH} ---")

    elif args.step == "train":
        print(f"--- Starting: Model Training for '{args.model.upper()}' ---")
        if not os.path.exists(config.FEATURE_PATH):
            print("Error: Feature file not found. Please run 'create_features' first.")
            return
        train_model(model_name=args.model)
        model_path = config.get_model_path(args.model)
        print(f"--- Finished: Model saved to {model_path} ---")

    elif args.step == "predict":
        print(f"--- Starting: Inference using '{args.model.upper()}' model ---")
        model_path = config.get_model_path(args.model)
        if not os.path.exists(model_path):
            print(f"Error: Trained model for '{args.model}' not found. Please run 'train --model {args.model}' first.")
            return

        soh_prediction = predict_soh(args.file_path, args.model, args.soc)
        if soh_prediction is not None:
            print(f"\nPredicted State of Health (SoH): {soh_prediction:.2%}")
        print("--- Finished: Inference ---")


if __name__ == "__main__":
    main()

