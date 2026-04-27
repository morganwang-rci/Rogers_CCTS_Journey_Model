import logging
from typing import Dict, List

import pandas as pd

from .config import Config
from .data_loader import DataLoader
from .relevancy import RelevancyChecker
from .analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class CCTSAnalysisPipeline:
    """Orchestrator for the CCTS analysis pipeline."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.relevancy_checker = RelevancyChecker(self.config)
        self.analyzer = StatisticalAnalyzer(self.config)

    def run(self) -> Dict[str, object]:
        logger.info("Starting CCTS analysis pipeline")

        df_ccts = self.data_loader.load_ccts_data()
        account_numbers = self._collect_account_numbers(df_ccts)

        df_chat = self.data_loader.load_chat_interactions(account_numbers)
        df_voice = self.data_loader.load_voice_interactions(account_numbers)
        df_all_interactions = self._prepare_interactions(df_chat, df_voice)

        if df_all_interactions.empty:
            logger.warning("No prior interactions were found for the configured accounts.")
            return self._empty_result()

        df_queue = self.data_loader.load_queue_hierarchy()

        df_interactions = self._merge_interactions_with_ccts(df_ccts, df_all_interactions)
        if df_interactions.empty:
            logger.warning("No interactions could be matched to CCTS complaints.")
            return self._empty_result()

        df_interactions = self.relevancy_checker.process_interactions_relevancy(df_interactions)

        stats = self.analyzer.calculate_basic_stats(df_ccts, df_interactions)
        df_history = self.analyzer.analyze_interaction_history(df_interactions)
        diagnostics = self.analyzer.analyze_departments_and_tiers(df_interactions, df_queue)

        plots = self.analyzer.create_plots(
            {
                "interaction_counts": df_history,
                "departments": diagnostics["departments"],
            }
        )

        logger.info("Completed CCTS analysis pipeline")
        return {
            "statistics": stats,
            "interaction_history": df_history,
            "diagnostics": diagnostics,
            "plots": plots,
        }

    def _collect_account_numbers(self, df_ccts: pd.DataFrame) -> List[str]:
        return (
            df_ccts["attr_account_number"]
            .dropna()
            .astype(str)
            .str.replace("-", "", regex=False)
            .unique()
            .tolist()
        )

    def _prepare_interactions(
        self, df_chat: pd.DataFrame, df_voice: pd.DataFrame
    ) -> pd.DataFrame:
        df_all = pd.concat([df_chat, df_voice], ignore_index=True)
        if df_all.empty:
            return pd.DataFrame()

        df_all["attr_account_number"] = (
            df_all["attr_account_number"].astype(str).str.replace("-", "", regex=False)
        )
        df_all["calendar_date"] = pd.to_datetime(df_all["calendar_date"], errors="coerce")
        return df_all.dropna(subset=["attr_account_number", "calendar_date"]).reset_index(drop=True)

    def _merge_interactions_with_ccts(
        self, df_ccts: pd.DataFrame, df_interactions: pd.DataFrame
    ) -> pd.DataFrame:
        df_ccts = df_ccts.copy()
        df_ccts["attr_account_number"] = (
            df_ccts["attr_account_number"].astype(str).str.replace("-", "", regex=False)
        )
        df_ccts["Email Date"] = pd.to_datetime(df_ccts["Email Date"], errors="coerce")

        df_merged = df_interactions.merge(
            df_ccts[
                ["Case Number", "attr_account_number", "Customer Verbatim", "Email Date"]
            ],
            on="attr_account_number",
            how="inner",
            suffixes=("", "_ccts"),
        )

        if df_merged.empty:
            return df_merged

        df_merged = df_merged[df_merged["calendar_date"] <= df_merged["Email Date"]].copy()
        return df_merged.reset_index(drop=True)

    def _empty_result(self) -> Dict[str, object]:
        return {
            "statistics": {
                "total_ccts": 0,
                "ccts_with_interactions": 0,
                "percentage_with_interactions": 0.0,
                "ccts_with_relevant_interactions": 0,
                "percentage_with_relevant": 0.0,
                "avg_relevant_interactions": 0.0,
            },
            "interaction_history": pd.DataFrame(),
            "diagnostics": {
                "departments": pd.DataFrame(),
                "tiers_ccts": pd.DataFrame(),
                "tiers_interactions": pd.DataFrame(),
                "filtered_data": pd.DataFrame(),
            },
            "plots": {},
        }


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    setup_logging()
    pipeline = CCTSAnalysisPipeline()
    result = pipeline.run()
    logger.info("Pipeline returned %d top-level keys", len(result))
