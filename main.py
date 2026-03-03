"""
AI AutoML Pipeline - Main Entry Point
"""

import argparse
import logging
import logging.handlers
import os

from src.pipeline import run_pipeline


def configure_logging() -> None:
    """
    Configure logging for the application.

    Sets up both console and file handlers with a consistent format:
    - Console output for immediate feedback
    - File output to logs/app.log for persistence
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Define logging format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (logs/app.log)
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def main() -> None:
    """Main entry point for the AI AutoML pipeline."""
    # Configure logging before anything else
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="AI AutoML Pipeline")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column")

    args = parser.parse_args()

    logger.info("Starting AI AutoML Pipeline")
    logger.info(f"Input file: {args.file}")
    logger.info(f"Target column: {args.target}")

    try:
        run_pipeline(file_path=args.file, target=args.target)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
