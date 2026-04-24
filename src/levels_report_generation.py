import logging
import sys
from pathlib import Path

import pandas as pd
# from openai import AzureOpenAI
import subprocess

# Install the openai package if not already installed
from openai import AzureOpenAI

from report_generation import (
    ConfigManager,
    InteractionAnalyzer,
    save_analysis_results,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # handlers=[
    #     logging.StreamHandler(),
    #     logging.FileHandler("level_reports.log"),
    # ],
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def levels_report_generation():
    """Main entry point for the analysis pipeline."""
    try:
        logger.info("=" * 80)
        logger.info("Starting Level Reports Generation analysis")
        logger.info("Step 1: Report Generation")
        logger.info("=" * 80)

        # Load configurations
        logger.info("Loading configurations...")
        azure_config = ConfigManager.load_azure_config()
        data_config = ConfigManager.load_data_config()
        processing_config = ConfigManager.load_processing_config()

        # Initialize Azure OpenAI client
        logger.info("Initializing Azure OpenAI client...")
        client = AzureOpenAI(
            api_key=azure_config.api_key,
            api_version=azure_config.api_version,
            azure_endpoint=azure_config.azure_endpoint,
        )

        # Load data
        logger.info(f"Loading pickle file from {data_config.pickle_file_path}")
        df = pd.read_pickle(data_config.pickle_file_path)
        logger.info(f"Loaded {len(df)} interactions from pickle file")

        # Filter for relevancy
        df_relevancy = df[df["relevancy"] == "yes"].copy()
        logger.info(f"Filtered to {len(df_relevancy)} relevant interactions")

        if len(df_relevancy) == 0:
            logger.warning("No relevant interactions found in data")
            return

        # Initialize analyzer
        analyzer = InteractionAnalyzer(client, processing_config)

        # Get unique cases
        cases = df_relevancy["Case Number"].unique()
        logger.info(f"Processing {len(cases)} unique cases")

        conversation_dir = Path(data_config.conversation_output_dir)
        journey_dir = Path(data_config.journey_output_dir)

        successful_cases = 0
        failed_cases = 0

        for case_num, case_number in enumerate(cases, 1):
            try:
                df_case = df_relevancy[df_relevancy["Case Number"] == case_number].copy()
                file_number = df_case.iloc[0]["File Number"]
                account_number = df_case.iloc[0]["attr_account_number"]

                logger.info(
                    f"[{case_num}/{len(cases)}] Processing case {case_number} "
                    f"({file_number}) for account {account_number}"
                )

                # Analyze interactions
                results = analyzer.analyze_all_interactions(df_case)

                if not results:
                    logger.warning(f"No results generated for case {case_number}")
                    failed_cases += 1
                    continue

                # Generate summary
                logger.info(f"Generating journey summary for case {case_number}")
                summary = analyzer.generate_summary_report(results, case_number)

                # Save results
                interaction_path, summary_path = save_analysis_results(
                    results, summary, case_number, conversation_dir, journey_dir
                )

                logger.info(
                    f"Case {case_number} analysis completed successfully. "
                    f"Results saved to:\n  - {interaction_path}\n  - {summary_path}"
                )

                successful_cases += 1

            except Exception as e:
                logger.error(f"Error processing case {case_number}: {str(e)}", exc_info=True)
                failed_cases += 1
                continue

        # Final summary
        logger.info("=" * 80)
        logger.info(
            f"Analysis pipeline completed. "
            f"Successful: {successful_cases}, Failed: {failed_cases}, "
            f"Total: {len(cases)}"
        )
        logger.info("=" * 80)

        if failed_cases > 0:
            logger.warning(f"{failed_cases} cases failed. Check logs for details.")
            sys.exit(1)

    except KeyError as e:
        logger.error(f"Configuration error: {str(e)}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main pipeline: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    levels_report_generation()



