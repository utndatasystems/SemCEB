import sys
import pandas as pd

from semceb.algorithms.helpers import get_dict_memory_usage, get_sample_memory_usage
from semceb.algorithms.interface import AlgorithmInterface
import lotus.settings
from semceb.queries.query_specification import QuerySpecification
from semceb.queries.template_parser import QueryTemplatePartType
from semceb.algorithms.cardinality_estimate import CardinalityEstimate, CardinalityEstimateKind


class ExtrapolatedSampling(AlgorithmInterface):
    """Algorithm based on extrapolation sampling."""

    def __init__(self, name: str, version: str):
        """Initialize the sampling algorithm and prepare cost tracking."""
        self.name = name
        self.version = version

        self.model = None
        self.reset_cost_stats()

    def get_memory_consumption(self) -> int:
        """
        Return the memory storage footprint of metadata structures used by the cardinality estimation algorithm in bytes.

        For example, if the algorithm holds a sample of the data, this method should return the size of that sample in bytes.
        """
        return self.memory_consumption

    def get_cost_stats(self) -> dict:
        """
        Return the accumulated cost statistics for cardinality estimation.
        """
        return self.cost_stats

    def reset_cost_stats(self) -> None:
        """Reset tracked algorithm cost."""
        self.cost_stats = {
            "usd": 0,
            "llm_calls": 0,
            "tokens": 0
            }

        if self.model is not None:
            self.model.reset_stats()
            # Prepare lotus for next run to determine costs
            lotus.settings.configure(lm=self.model)

    def preparation(self, data_dfs: dict[str, pd.DataFrame], algorithm_kwargs: dict) -> None:
        """Prepare the algorithm before execution.

        This method should collect and store all information required for selectivity estimation.
        Specifially, implementations are expected to take ownership of the provided dataframes.

        During execution, the algorithm will only receive the dataset name(s)
        and column name(s) needed to perform the selectivity estimate.
        """

        self.data_rows = {
            name: df.shape[0]
            for name, df in data_dfs.items()
        }

        sampling_frac = algorithm_kwargs.get("sampling_frac", -1)
        self._validate_sampling_frac(sampling_frac)

        model_name = algorithm_kwargs.get("model_name", None)
        self._validate_model_name(model_name)

        self._initialize_model(
            model_name=model_name,
            system_prompt=algorithm_kwargs.get("system_prompt", None),
        )

        self.data_sample = self._create_data_samples(
            data_dfs=data_dfs,
            sampling_frac=sampling_frac,
        )

        self.memory_consumption = (
            get_dict_memory_usage(self.data_rows)
            + get_sample_memory_usage(self.data_sample)
            + sys.getsizeof(self.model)
        )

    def _validate_sampling_frac(self, sampling_frac: float) -> None:
        """Validate that the sample fraction is a positive fraction at most 1."""
        if not 0 < sampling_frac <= 1:
            raise ValueError("sampling_frac must be in the interval (0, 1].")

    def _validate_model_name(self, model_name: str | None) -> None:
        """Validate that a non-empty model name was provided."""
        if not model_name:
            raise ValueError("model_name must be a valid name of a model.")

    def _initialize_model(self, model_name: str, system_prompt: str | None) -> None:
        """Initialize and configure the Lotus LM backend used by the algorithm."""
        from lotus.cache import CacheConfig, CacheFactory, CacheType
        from lotus.models.lm import LM

        cache_config = CacheConfig(
            cache_type=CacheType.IN_MEMORY,
            max_size=1000,
        )
        cache = CacheFactory.create_cache(cache_config)

        self.model = LM(
            model=model_name,
            rate_limit=None,
            max_batch_size=64,
            cache=cache,
        )
        self.model.system_prompt = system_prompt

        lotus.settings.configure(
            lm=self.model,
            enable_cache=True,
        )

    def _create_data_samples(
        self,
        data_dfs: dict[str, pd.DataFrame],
        sampling_frac: float,
    ) -> dict[str, pd.DataFrame]:
        """Create a random sample of each dataset at the requested fraction."""
        data_sample: dict[str, pd.DataFrame] = {}

        for name, df in data_dfs.items():
            sample_df = df.sample(frac=sampling_frac, random_state=42)
            if sample_df.empty:
                raise ValueError(
                    f"Sample of dataframe '{name}' is empty. Increase sampling_frac or provide more data."
                )

            data_sample[name] = sample_df

        return data_sample

    def run(self, query_spec: QuerySpecification) -> CardinalityEstimate:
        """Run the algorithm and return the estimated output cardinality for the given query."""

        if len(query_spec.datasets) == 1:
            estimation = max(1, self._estimate_filter_cardinality(query_spec))
            return CardinalityEstimate(CardinalityEstimateKind.INT, value=estimation)
        elif len(query_spec.datasets) > 1:
            estimation = max(1, self._estimate_join_cardinality(query_spec))
            return CardinalityEstimate(CardinalityEstimateKind.INT, value=estimation)
        
        raise ValueError("Query must contain at least one dataset.")

    def _estimate_filter_cardinality(self, query_spec: QuerySpecification) -> int:
        """Estimate cardinality for a single-table filter query from the sample."""
        query_str = self._build_filter_query_str(query_spec)

        dataset_spec = query_spec.datasets[0]
        data_sample = self.data_sample[dataset_spec.table_ref]

        sample_estimation = data_sample.sem_filter(
            user_instruction=query_str,
        ).shape[0]

        cardinality_estimation = self._extrapolate_cardinality(
            sample_estimation=sample_estimation,
            sample_size=data_sample.shape[0],
            total_size=self.data_rows[dataset_spec.table_ref],
        )

        self._track_costs(data_sample)
        return cardinality_estimation

    def _estimate_join_cardinality(self, query_spec: QuerySpecification) -> int:
        """Estimate cardinality for a two-table join query from sampled joins."""
        dataset_spec_left, dataset_spec_right = query_spec.datasets
        data_left = self.data_sample[dataset_spec_left.table_ref]
        data_right = self.data_sample[dataset_spec_right.table_ref]

        query_str = self._format_join_query(
            query_spec=query_spec,
            dataset_side={
                dataset_spec_left.alias: "left",
                dataset_spec_right.alias: "right",
            },
        )

        sample_estimation = data_left.sem_join(
            data_right,
            query_str,
        ).shape[0]

        sample_pair_count = (
            self.data_sample[dataset_spec_left.table_ref].shape[0]
            * self.data_sample[dataset_spec_right.table_ref].shape[0]
        )
        total_pair_count = (
            self.data_rows[dataset_spec_left.table_ref]
            * self.data_rows[dataset_spec_right.table_ref]
        )

        cardinality_estimation = self._extrapolate_cardinality(
            sample_estimation=sample_estimation,
            sample_size=sample_pair_count,
            total_size=total_pair_count,
        )

        llm_cost_stats_left = self._get_costs(
            self.data_sample[dataset_spec_left.table_ref]
        )
        llm_cost_stats_right = self._get_costs(
            self.data_sample[dataset_spec_right.table_ref]
        )
        llm_cost_stats = llm_cost_stats_left
        llm_cost_stats["llm_calls"] *= llm_cost_stats_right["llm_calls"]

        self._add_cost(llm_cost_stats)
        return cardinality_estimation

    def _build_filter_query_str(self, query_spec: QuerySpecification) -> str:
        """Convert a parsed filter template into a LOTUS filter string."""
        query_str = ""

        for part in query_spec.filter_parsed.parts:
            if part.type == QueryTemplatePartType.TEXT:
                query_str += part.value
            elif part.type == QueryTemplatePartType.COLUMN_REF:
                query_str += f"{{{part.value.column_name}}}"

        return query_str

    def _format_join_query(
        self,
        query_spec: QuerySpecification,
        dataset_side: dict[str, str],
    ) -> str:
        """Convert a join query template into a Lotus-compatible join string."""
        query_parts: list[str] = []
        current_text: list[str] = []
        current_column: list[str] | None = None

        for char in query_spec.filter:
            if char == "{":
                if current_column is not None:
                    raise ValueError("Nested '{' is not allowed.")

                if current_text:
                    query_parts.append("".join(current_text))
                    current_text = []

                current_column = []

            elif char == "}":
                if current_column is None:
                    raise ValueError("Found '}' without matching '{'.")

                column_reference = "".join(current_column).strip()
                query_parts.append(
                    self._format_join_column_reference(
                        column_reference=column_reference,
                        dataset_side=dataset_side,
                        query_spec=query_spec,
                    )
                )
                current_column = None

            else:
                if current_column is not None:
                    current_column.append(char)
                else:
                    current_text.append(char)

        if current_column is not None:
            raise ValueError("Found '{' without matching '}'.")

        if current_text:
            query_parts.append("".join(current_text))

        return "".join(query_parts)

    def _format_join_column_reference(
        self,
        column_reference: str,
        dataset_side: dict[str, str],
        query_spec: QuerySpecification,
    ) -> str:
        """Format a dataset-qualified join column reference for Lotus."""
        if "." not in column_reference:
            raise ValueError(
                f"Invalid column reference '{column_reference}'. "
                "Expected format: '<dataset>.<column>'."
            )

        dataset_name, column_name = column_reference.split(".", maxsplit=1)

        if dataset_name not in dataset_side:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Expected one of: {query_spec.datasets}."
            )

        return f"{{{column_name}:{dataset_side[dataset_name]}}}"

    def _extrapolate_cardinality(
        self,
        sample_estimation: int,
        sample_size: int,
        total_size: int,
    ) -> int:
        """Scale sample cardinality to the full dataset using observed selectivity."""
        if sample_size == 0:
            return 0

        selectivity = sample_estimation / sample_size
        cardinality = selectivity * total_size

        return max(
            0,
            min(
                round(cardinality),
                total_size,
            ),
        )

    def _track_costs(self, data: pd.DataFrame) -> None:
        """Track the cost incurred by a semantic operation on the sample."""
        llm_cost_stats = self._get_costs(data)
        self._add_cost(llm_cost_stats)


    def _get_costs(self, data: pd.DataFrame) -> dict:
        """Return virtual (= without caching) cost stats for the last Lotus run."""
        return  {
            "usd": self.model.stats.virtual_usage.total_cost,
            "llm_calls": data.shape[0], # Calculation possible because no cascade
            "tokens": self.model.stats.virtual_usage.total_tokens
        }


    def _add_cost(self, cost_stats: dict) -> None:
        """Accumulate cost statistics from one semantic operation."""
        self.cost_stats["usd"] += cost_stats["usd"]
        self.cost_stats["llm_calls"] += cost_stats["llm_calls"]
        self.cost_stats["tokens"] += cost_stats["tokens"]
