import sys
import pandas as pd

from runner.algorithms.interface import AlgorithmInterface
import lotus.settings
from queries.query_specification import QuerySpecification
from queries.template_parser import QueryTemplatePartType


class ExtrapolatedSampling(AlgorithmInterface):
    """Algorithm based on extrapolation sampling."""

    def __init__(self, name: str, version: str):
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
        
        # Data set to evaluate
        self.data_rows = {}
        for name, df in data_dfs.items():
            self.data_rows[name] = df.shape[0]

        # Algorithm specific arguments
        sampling_frac = algorithm_kwargs.get("sampling_frac", -1)
        if not 0 < sampling_frac <= 1:
            raise ValueError("sampling_frac must be in the interval (0, 1].")
        
        model_name = algorithm_kwargs.get("model_name", None)
        if not model_name:
            raise ValueError("model_name must be a valid name of a model.")

        import lotus.settings
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
        self.model.system_prompt = algorithm_kwargs.get("system_prompt", None)
        lotus.settings.configure(
            lm=self.model,
            enable_cache=True,
        )        

        self.data_sample = {}
        for name, df in data_dfs.items():
            self.data_sample[name] = df.sample(frac=sampling_frac, random_state=42)
            if self.data_sample[name].empty:
                raise ValueError(
                    f"Sample of dataframe '{name}' is empty. Increase sampling_frac or provide more data."
                )

        self.memory_consumption = (
            sys.getsizeof(self.data_rows)
            + sys.getsizeof(self.data_sample)
            + sys.getsizeof(self.model)
        )

    def run(self, query_spec: QuerySpecification) -> int:
        """Run the algorithm and return the estimated output cardinality for the given query."""

        # Filtering
        if len(query_spec.datasets) == 1:

            # Evaluate sample
            query_str = ""
            for part in query_spec.filter_parsed.parts:
                if part.type == QueryTemplatePartType.TEXT:
                    query_str += part.value
                if part.type == QueryTemplatePartType.COLUMN_REF:
                    query_str += f"{{{part.value.column_name}}}"

            dataset_spec = query_spec.datasets[0]

            sample_estimation = self.data_sample[dataset_spec.table_ref].sem_filter(
                user_instruction=query_str,
            ).shape[0]

            # Extrapolate to estimate
            selectivity_estimation = (
                sample_estimation / self.data_sample[dataset_spec.table_ref].shape[0] * self.data_rows[dataset_spec.table_ref]
            )

            # Track costs
            llm_cost_stats = self._get_costs(self.data_sample[dataset_spec.table_ref])
            self._add_cost(llm_cost_stats)
            
            selectivity_estimation = max(0, min(round(selectivity_estimation), self.data_rows[dataset_spec.table_ref]))

        # Joining
        elif len(query_spec.datasets) > 1:

            # Evaluate sample

            dataset_spec_left, dataset_spec_right = query_spec.datasets
            data_left = self.data_sample[dataset_spec_left.table_ref]
            data_right = self.data_sample[dataset_spec_right.table_ref]
            
            dataset_side = {
                dataset_spec_left.alias: "left",
                dataset_spec_right.alias: "right",
            }

            query_parts = []
            current_text = []
            current_column = None

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

                    side = dataset_side[dataset_name]
                    query_parts.append(f"{{{column_name}:{side}}}")

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

            query_str = "".join(query_parts)

            sample_estimation = data_left.sem_join(
                data_right,
                query_str,
            ).shape[0]

            # Extrapolate to estimate
            n_left_sample = self.data_sample[dataset_spec_left.table_ref].shape[0]
            n_right_sample = self.data_sample[dataset_spec_right.table_ref].shape[0]

            n_left_total = self.data_rows[dataset_spec_left.table_ref]
            n_right_total = self.data_rows[dataset_spec_right.table_ref]

            sample_pair_count = n_left_sample * n_right_sample
            total_pair_count = n_left_total * n_right_total

            join_selectivity = sample_estimation / sample_pair_count

            selectivity_estimation = join_selectivity * total_pair_count

            # Track costs
            llm_cost_stats_left = self._get_costs(self.data_sample[dataset_spec_left.table_ref])
            llm_cost_stats_right = self._get_costs(self.data_sample[dataset_spec_right.table_ref]) # to get llm calls
            llm_cost_stats = llm_cost_stats_left
            llm_cost_stats["llm_calls"] *= llm_cost_stats_right["llm_calls"]
            self._add_cost(llm_cost_stats)

        return selectivity_estimation


    def _get_costs(self, data: pd.DataFrame) -> dict:
        """Return virtual (= without caching) cost stats."""
        return  {
            "usd": self.model.stats.virtual_usage.total_cost,
            "llm_calls": data.shape[0], # Calculation possible because no cascade
            "tokens": self.model.stats.virtual_usage.total_tokens
        }


    def _add_cost(self, cost_stats: dict) -> None:
        """Add cost amount to tracked algorithm cost."""
        self.cost_stats["usd"] += cost_stats["usd"]
        self.cost_stats["llm_calls"] += cost_stats["llm_calls"]
        self.cost_stats["tokens"] += cost_stats["tokens"]
