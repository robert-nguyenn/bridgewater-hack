"""Core pydantic schemas. THE CONTRACT.

Every other module imports from here. Do not change these schemas without
explicit team consensus.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import date
from enum import Enum


class PolicyType(str, Enum):
    MONETARY = "monetary"
    FISCAL = "fiscal"
    TRADE = "trade"
    REGULATORY = "regulatory"
    GEOPOLITICAL = "geopolitical"


class VariableType(str, Enum):
    PRICE = "price"                    # rates, FX, commodity prices, equity
    FUNDAMENTAL = "fundamental"        # margins, earnings, CPI
    PROBABILITY = "probability"        # bounded 0 to 1
    SENTIMENT = "sentiment"            # text derived, standardized
    DISTRIBUTIONAL = "distributional"  # Kalshi moments, implied vol


class EstimatorType(str, Enum):
    EVENT_STUDY = "event_study"
    LEVEL_REGRESSION = "level_regression"
    CROSS_SECTION = "cross_section"
    ANALOG_RETRIEVAL = "analog_retrieval"
    SVAR_LOOKUP = "svar_lookup"
    KALSHI_CONDITIONAL = "kalshi_conditional"


class StructuredPolicy(BaseModel):
    """Output of the policy parser. The structured representation of user input."""
    raw_input: str
    policy_type: PolicyType
    subject: str                     # "Chinese semiconductors", "fed funds rate"
    magnitude: float                 # e.g., 0.25 for 25% tariff, -50 for 50bp cut
    magnitude_unit: str              # "percent", "basis_points", "probability"
    direction: Literal["positive", "negative", "bidirectional"]
    horizon_days: int                # time frame the user is asking about
    effective_date: Optional[date] = None
    additional_context: Optional[str] = None


class HistoricalEpisode(BaseModel):
    name: str                        # human readable
    date: date
    magnitude: float
    notes: Optional[str] = None
    adversarial_critique: Optional[str] = None  # populated by the adversary agent,
                                                # flags what makes this analog a weak
                                                # comparison for the current scenario


class Confounder(BaseModel):
    name: str
    mechanism: str
    proxy_variable: str              # FRED series id or constructed series name
    handling: Literal["include_covariate", "sample_restriction", "identification"]
    expected_direction: Optional[str] = None


class Hypothesis(BaseModel):
    """A fully specified empirical test proposed by a specialist agent."""
    hypothesis_id: str               # unique id for tracking
    proposed_by: str                 # agent name
    channel_id: str                  # must match an entry in channel_catalog.yaml

    # What is being shocked
    shock_variable: str
    shock_type: VariableType
    shock_source_hints: list[str]    # data source hints for the loader

    # What is being measured
    response_variable: str
    response_type: VariableType
    response_source_hints: list[str]

    # How to estimate
    estimator: EstimatorType
    specification_params: dict       # estimator specific (window size, lags, etc)

    # Historical episodes to use
    historical_episodes: list[HistoricalEpisode]

    # Controls
    covariates: list[str]
    confounders: list[Confounder] = []

    # Metadata
    expected_sign: Literal["positive", "negative", "ambiguous"]
    economic_rationale: str
    citations: list[str] = []        # speech passages, papers, etc


class EstimateRange(BaseModel):
    point: float
    low: float
    high: float
    unit: str


class ConfidenceBreakdown(BaseModel):
    statistical: float               # 0 to 1, from SE relative to coef
    sample: float                    # 0 to 1, from n
    cross_method: float              # 0 to 1, agreement across methods
    regime: float                    # 0 to 1, analog regime similarity
    overall: float                   # weighted aggregate


class MethodEstimate(BaseModel):
    """Output of a single estimator for a single hypothesis."""
    method: EstimatorType
    coefficient: Optional[float]
    standard_error: Optional[float]
    sample_size: Optional[int]
    r_squared: Optional[float]
    passed: bool                     # did the estimate run successfully
    notes: Optional[str] = None
    plot_path: Optional[str] = None  # matplotlib figure, relative to project root


class EdgeObject(BaseModel):
    """A single edge in the impact graph."""
    source_node: str
    target_node: str
    wave: int                        # 1, 2, or 3

    elasticity: EstimateRange
    confidence: ConfidenceBreakdown
    lag_days: int
    causal_share: Optional[float]    # fraction of variance explained
    method_estimates: list[MethodEstimate]
    confounders_tested: list[Confounder]
    caveats: list[str]

    hypothesis_ids: list[str]        # traceability back to hypotheses
    is_first_link: bool = False      # true if source is the scenario event itself.
                                     # first link edges are LLM reasoning and analog
                                     # driven, not empirically estimated. Confidence
                                     # and UI treatment differ.


class NodeObject(BaseModel):
    """A single node in the impact graph."""
    node_id: str
    label: str
    variable_type: VariableType
    wave: int
    current_level: Optional[float]
    projected_level: Optional[float]
    projected_range: Optional[tuple[float, float]]
    data_source: Optional[str]


class ReviewFlag(BaseModel):
    """One issue raised by the review agent about an edge or the full map."""
    severity: Literal["info", "warning", "error"]
    target: str                      # edge_id or "global"
    category: str                    # sign_mismatch, sample_size, r2, plot_vs_claim, invented_number
    message: str


class ImpactMap(BaseModel):
    """The full output of one run of the pipeline."""
    policy: StructuredPolicy
    nodes: list[NodeObject]
    edges: list[EdgeObject]
    historical_analogs: list[dict]   # results from analog retriever
    kalshi_signals: list[dict]       # results from Kalshi specialist
    data_availability_report: dict
    generation_metadata: dict        # timestamps, model versions, run id
    review_flags: list[ReviewFlag] = []


class DataResponse(BaseModel):
    """What the data loader returns to the empirics layer."""
    success: bool
    dataframe_dict: dict             # serialized polars frames by role
    missing: list[str]
    warnings: list[str]
