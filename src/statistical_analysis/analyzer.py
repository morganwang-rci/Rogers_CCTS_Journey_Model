import logging
from typing import Dict, Any

import pandas as pd
import plotly.express as px

from .config import Config

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Performs statistical analysis on the processed data."""

    def __init__(self, config: Config):
        """Initialize StatisticalAnalyzer."""
        self.config = config

    def calculate_basic_stats(self, df_ccts: pd.DataFrame, df_interactions: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics."""
        stats = {}
        total_ccts = len(df_ccts["Case Number"].unique())
        stats["total_ccts"] = total_ccts

        ccts_with_interactions = len(df_interactions["Case Number"].unique())
        stats["ccts_with_interactions"] = ccts_with_interactions
        stats["percentage_with_interactions"] = (
            (ccts_with_interactions / total_ccts * 100) if total_ccts else 0.0
        )

        df_relevant = df_interactions[df_interactions["relevancy"] == "yes"]
        relevant_ccts = len(df_relevant["Case Number"].unique())
        stats["ccts_with_relevant_interactions"] = relevant_ccts
        stats["percentage_with_relevant"] = (
            (relevant_ccts / ccts_with_interactions * 100)
            if ccts_with_interactions
            else 0.0
        )

        df_count = df_relevant.groupby("Case Number").agg(
            interaction_count=("relevancy", "size"),
            relevant_interaction_count=("relevancy", lambda x: x[x == "yes"].count()),
        )
        stats["avg_relevant_interactions"] = (
            df_count["relevant_interaction_count"].mean() if not df_count.empty else 0.0
        )

        logger.info("Calculated basic statistics")
        return stats

    def analyze_interaction_history(self, df_interactions: pd.DataFrame) -> pd.DataFrame:
        """Analyze interaction history for relevant interactions."""
        logger.info("Analyzing interaction history")
        df_relevant = df_interactions[df_interactions["relevancy"] == "yes"].copy()

        def avg_date_diff(group):
            sorted_dates = pd.to_datetime(group).sort_values()
            diffs = sorted_dates.diff().dropna()
            return diffs.mean().days if not diffs.empty else None

        df_history = df_relevant.groupby(["Case Number", "Email Date"]).agg(
            interaction_count=("full_transcript", "size"),
            transcripts_list=("full_transcript", lambda x: list(x)),
            sum_interaction_time=("talk_time", "sum"),
            avg_interaction_diff=("calendar_date", avg_date_diff),
            interaction_dates_list=("calendar_date", lambda x: list(x.sort_values())),
        ).reset_index()

        df_history["first_interaction_to_ccts_diff"] = (
            pd.to_datetime(df_history["Email Date"]) -
            pd.to_datetime(df_history["interaction_dates_list"].apply(lambda x: x[0]))
        ).dt.days

        df_history["last_interaction_to_ccts_diff"] = (
            pd.to_datetime(df_history["Email Date"]) -
            pd.to_datetime(df_history["interaction_dates_list"].apply(lambda x: x[-1]))
        ).dt.days

        self._bucket_interactions(df_history)
        logger.info("Completed interaction history analysis")
        return df_history

    def _bucket_interactions(self, df: pd.DataFrame) -> None:
        df.loc[df["interaction_count"] > 20, "interaction_count_bucket"] = "20+"
        df.loc[(df["interaction_count"] > 10) & (df["interaction_count"] <= 20), "interaction_count_bucket"] = "11-20"
        df.loc[(df["interaction_count"] > 5) & (df["interaction_count"] <= 10), "interaction_count_bucket"] = "6-10"
        df.loc[(df["interaction_count"] > 1) & (df["interaction_count"] <= 5), "interaction_count_bucket"] = "2-5"
        df.loc[df["interaction_count"] == 1, "interaction_count_bucket"] = "1"

        for col, bucket_col in [
            ("first_interaction_to_ccts_diff", "first_interaction_to_ccts_bucket"),
            ("last_interaction_to_ccts_diff", "last_interaction_to_ccts_bucket"),
        ]:
            df.loc[df[col] > 30, bucket_col] = "30+"
            df.loc[(df[col] > 7) & (df[col] <= 30), bucket_col] = "8-30"
            df.loc[(df[col] > 1) & (df[col] <= 7), bucket_col] = "2-7"
            df.loc[df[col] == 1, bucket_col] = "1"
            df.loc[df[col] == 0, bucket_col] = "0"

        df.loc[df["sum_interaction_time"] > 10800, "sum_interaction_count_bucket"] = "> 3 hr."
        df.loc[(df["sum_interaction_time"] > 7200) & (df["sum_interaction_time"] <= 10800), "sum_interaction_count_bucket"] = "2 to 3 hr."
        df.loc[(df["sum_interaction_time"] > 3600) & (df["sum_interaction_time"] <= 7200), "sum_interaction_count_bucket"] = "1 to 2 hr."
        df.loc[(df["sum_interaction_time"] > 1800) & (df["sum_interaction_time"] <= 3600), "sum_interaction_count_bucket"] = "30 to 60 min."
        df.loc[(df["sum_interaction_time"] > 900) & (df["sum_interaction_time"] <= 1800), "sum_interaction_count_bucket"] = "15 to 30 min."
        df.loc[(df["sum_interaction_time"] > 0) & (df["sum_interaction_time"] <= 900), "sum_interaction_count_bucket"] = "< 15 min."

    def analyze_departments_and_tiers(self, df_interactions: pd.DataFrame, df_queues: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        logger.info("Analyzing departments and tiers")
        df_merged = df_interactions.merge(df_queues, on="queue_name", how="inner")
        df_filtered = df_merged[
            (df_merged["effective_from"] <= df_merged["calendar_date"]) &
            (df_merged["calendar_date"] <= df_merged["effective_to"])
        ]
        df_filtered = self._clean_department_names(df_filtered)

        df_dep_ccts = (
            df_filtered.groupby("dept_name")
            .agg(CCTS_Count=("Case Number", lambda x: len(set(x))))
            .reset_index()
            .sort_values(by="CCTS_Count", ascending=False)
        )

        df_filtered.loc[df_filtered["lifecycle_name"] != "Escalations", "lifecycle_name"] = "Tier 1"
        df_filtered.loc[df_filtered["lifecycle_name"] == "Escalations", "lifecycle_name"] = "Tier 2"

        df_tiers_ccts = (
            df_filtered.groupby("lifecycle_name")
            .agg(CCTS_Count=("Case Number", lambda x: len(set(x))))
            .reset_index()
            .sort_values(by="CCTS_Count", ascending=False)
        )

        df_tiers_interactions = (
            df_filtered.groupby("lifecycle_name")
            .size()
            .sort_values(ascending=False)
            .to_frame("Interaction Count")
            .reset_index()
        )

        logger.info("Completed department and tier analysis")
        return {
            "departments": df_dep_ccts,
            "tiers_ccts": df_tiers_ccts,
            "tiers_interactions": df_tiers_interactions,
            "filtered_data": df_filtered,
        }

    def _clean_department_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["dept_name"] = df["dept_name"].str.strip()
        mappings = {
            "SoHo Service": "Business Care",
            "R4B Commercial Service": "Business Care",
            "R4B Tech Support Cable": "Business Tech Support",
            "R4B Tech Support Wireless": "Business Tech Support",
            "Credit Operations Chat": "Credit Ops",
            "Credit Operations": "Credit Ops",
            "Credit Operations Solution Support": "Credit Ops Tier 2",
            "Care Service": "Frontline Care",
            "Ignite Service": "Frontline Care",
            "RPP Service": "Frontline Care",
            "Wireless Service": "Frontline Care",
            "Consolidated Service": "Frontline Care",
            "Customer Porting Centre": "Frontline Care",
            "Connect for Success": "Frontline Care",
            "Wireless Tier 1": "Frontline Tech Support",
            "Connected Home Tier 1": "Frontline Tech Support",
            "Prepaid": "Prepaid (Should be minimal)",
            "Customer Inside Sales": "Telesales",
            "Consumer Outbound": "Telesales",
            "National Inside Sales": "Telesales",
            "Customer Escalation": "Tier 2 Care",
            "Solution Support Desk": "Tier 2 Care",
            "Retail Support Group": "Tier 2 Care",
            "Connected Home Tier 2": "Tier 2 Tech Support",
            "Wireless Tier 2": "Tier 2 Tech Support",
            "Office of the President": "OOP",
            "TBD missing information Internal": "Other",
        }
        for old_name, new_name in mappings.items():
            df.loc[df["dept_name"] == old_name, "dept_name"] = new_name
        return df

    def create_plots(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Creating plots")
        plots = {}

        if "interaction_counts" in data:
            df = data["interaction_counts"]
            if not df.empty:
                bucket_counts = (
                    df.groupby("interaction_count_bucket")["Case Number"]
                    .nunique()
                    .reset_index(name="Number of CCTS Cases")
                )
                bucket_counts = bucket_counts.sort_values(
                    key=lambda x: x.map({"1": 0, "2-5": 1, "6-10": 2, "11-20": 3, "20+": 4})
                )
                fig = px.bar(
                    bucket_counts,
                    x="interaction_count_bucket",
                    y="Number of CCTS Cases",
                    text="Number of CCTS Cases",
                    title=f"Interactions Count Prior to {self.config.analysis_period} CCTSs",
                    category_orders={"interaction_count_bucket": ["1", "2-5", "6-10", "11-20", "20+"]},
                    color="interaction_count_bucket",
                    width=900,
                    height=600,
                    labels={"interaction_count_bucket": "Number of Interactions"},
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                fig.update_yaxes(showticklabels=False)
                plots["interaction_counts"] = fig

        if "departments" in data:
            df = data["departments"]
            if not df.empty:
                fig = px.bar(
                    df,
                    x="dept_name",
                    y="CCTS_Count",
                    text="CCTS_Count",
                    title="Department Names Customers Connected to",
                    color="dept_name",
                    width=900,
                    height=600,
                )
                fig.update_layout(showlegend=False)
                fig.update_traces(textposition="outside")
                plots["departments"] = fig

        logger.info("Created %d plots", len(plots))
        return plots
