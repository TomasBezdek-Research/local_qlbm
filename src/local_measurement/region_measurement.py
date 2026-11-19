"""
Problem-independent utilities for estimating probability within a region.

The module defines a generic interface between a quantum application and
multiple region-estimation methods. Depending on the selected method, an
estimation problem may provide:

- a factory for allocating the required quantum arguments,
- a state-preparation function,
- a flag or phase oracle for quantum amplitude estimation, or
- a classical classifier and measurement function for classical sampling.

The concrete estimation algorithm is selected through
`RegionMeasurementMethod`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, log2, pi, sin
from typing import Any

from qrisp import QuantumBool, QuantumFloat

QArgs = Sequence[Any]
QArgsFactory = Callable[[], QArgs]
TargetFactory = Callable[[], QuantumBool]

StateFunction = Callable[..., None]
FlagOracle = Callable[..., None]
PhaseOracle = Callable[..., None]
MeasurementDict = Mapping[Any, float]
SamplingMeasurement = Callable[..., MeasurementDict]


class RegionMeasurementMethod(str, Enum):
    """Supported region-probability estimation methods."""

    OBPA = "obpa"
    QAE = "qae"
    IQAE = "iqae"
    CLASSICAL_SAMPLING = "classical_sampling"


@dataclass(frozen=True)
class RegionEstimate:
    """
    Result of a region-probability estimation.

    Args:
        probability: Estimated probability mass inside the region.
        lower_bound: Optional lower bound on the estimate.
        upper_bound: Optional upper bound on the estimate.
        method: Estimation method used to obtain the result.
        statistics: Optional method-specific estimation statistics.
    """

    probability: float

    lower_bound: float | None = None
    upper_bound: float | None = None

    method: RegionMeasurementMethod | None = None

    statistics: RegionEstimateStatistics | None = None


@dataclass(frozen=True)
class RegionEstimateStatistics:
    """
    Method-specific statistics associated with a region estimate.

    Args:
        shots: Number of measurement shots used.
        requested_eps: Requested estimation accuracy.
        requested_alpha: Requested failure probability.
        confidence_level: Confidence level associated with the estimate.
        theoretical_error_bound: Theoretical error bound of the method.
        qae_precision: Precision parameter used for canonical QAE.
        sample_variance: Variance of the classical sampling estimate.
        sample_standard_error: Standard error of the sampling estimate.
        metadata: Additional method-specific statistics.
    """

    shots: int | None = None

    requested_eps: float | None = None
    requested_alpha: float | None = None

    confidence_level: float | None = None

    theoretical_error_bound: float | None = None

    qae_precision: int | None = None

    sample_variance: float | None = None
    sample_standard_error: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionEstimationInput:
    """
    Problem definition for region-probability estimation.

    The required components depend on the selected estimation method.
    Quantum amplitude-estimation methods use region oracles, while classical
    sampling uses a measurement function together with a classical classifier.

    Args:
        qargs:
            Factory returning the quantum arguments required by the problem.

        state_function:
            Callable preparing the state whose region probability is estimated.
            It receives the quantum arguments produced by `qargs`.

        flag_oracle:
            Optional oracle that flips a target qubit for states inside
            the region.

        phase_oracle:
            Optional oracle that applies a phase flip to states inside
            the region.

        classical_classifier:
            Optional callable determining whether a measured sample belongs
            to the region.

        target:
            Optional factory creating a target qubit when it is not already
            included in the arguments returned by `qargs`.

        sampling_measurement:
            Optional callable measuring the prepared state for classical
            sampling. Runtime measurement options such as `shots` and
            `backend` are passed to this callable.

        metadata:
            Additional user-defined information associated with the problem.
    """

    qargs: QArgsFactory
    state_function: StateFunction

    flag_oracle: FlagOracle | None = None

    @property
    def nonnone_flag_oracle(self) -> FlagOracle:
        """Return the flag oracle, raising an error if none is configured."""
        if self.flag_oracle is None:
            raise ValueError("flag_oracle is None")
        return self.flag_oracle

    phase_oracle: PhaseOracle | None = None

    @property
    def nonnone_phase_oracle(self) -> PhaseOracle:
        """Return the phase oracle, raising an error if none is configured."""
        if self.phase_oracle is None:
            raise ValueError("phase_oracle is None")
        return self.phase_oracle

    classical_classifier: Callable[[Any], bool] | None = None

    target: TargetFactory | None = None

    @property
    def nonnone_target(self) -> TargetFactory:
        """Return the target factory, raising an error if none is configured."""
        if self.target is None:
            raise ValueError("target is None")
        return self.target

    sampling_measurement: Callable[..., Any] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_qargs(self) -> QArgsFactory:
        """
        Return a quantum-argument factory that includes the target qubit.

        If a separate target factory is configured, the created target qubit is
        appended as the final argument. Otherwise, the original `qargs` factory
        is returned unchanged.
        """
        if self.target is None:
            return self.qargs

        def factory() -> tuple[Any, ...]:
            args = tuple(self.qargs())
            target = self.nonnone_target()
            return (*args, target)

        return factory


@dataclass(frozen=True)
class EstimationOptions:
    """
    Runtime configuration for region-probability estimation.

    Args:
        method:
            Estimation method to use. If omitted, IQAE is used by default.

        eps:
            Target estimation accuracy for IQAE and for deriving the default
            canonical-QAE precision.

        alpha:
            Failure probability used by IQAE.

        qae_precision:
            Precision used by canonical QAE. If omitted, it is derived
            from `eps`.

        shots:
            Number of measurement shots.

        backend:
            Backend used for measurement.

        mes_kwargs:
            Additional measurement arguments passed to Qrisp.

        qae_target:
            Optional target quantum float used by canonical QAE. When
            provided, it takes precedence over `qae_precision`.
    """

    method: RegionMeasurementMethod | None = None

    eps: float = 0.01
    alpha: float = 0.05

    qae_precision: int | None = None

    @property
    def resolved_precision(self) -> int:
        """Return the explicit QAE precision or derive it from the target accuracy."""
        if self.qae_precision is not None:
            return self.qae_precision
        return max(1, ceil(log2(1 / self.eps)))

    shots: int | None = None
    backend: object | None = None

    mes_kwargs: dict[str, Any] = field(default_factory=dict)

    qae_target: QuantumFloat | None = None


def estimate_region_probability(
    problem: RegionEstimationInput,
    options: EstimationOptions | None = None,
    estimation_method: RegionMeasurementMethod | None = None,
) -> RegionEstimate:
    """
    Estimate the probability mass contained in a region.

    The requested estimation method is selected explicitly through
    `estimation_method`, through `options.method`, or defaults to IQAE.

    Args:
        problem:
            Region-estimation problem containing the state preparation and
            method-specific region information.

        options:
            Runtime configuration for the estimation. Default options are used
            when omitted.

        estimation_method:
            Optional method overriding the method specified in `options`.

    Returns:
        Estimated region probability and associated statistics.

    Raises:
        ValueError:
            If the selected estimation method is unsupported.
    """

    options = options or EstimationOptions()

    if estimation_method is not None:
        method = estimation_method
    elif options.method is not None:
        method = options.method
    else:
        method = RegionMeasurementMethod.IQAE

    if method == RegionMeasurementMethod.OBPA:
        return obpa_region_estimate(
            problem,
            shots=options.shots,
            backend=options.backend,
        )

    if method == RegionMeasurementMethod.QAE:
        return qae_region_estimate(problem, options)

    if method == RegionMeasurementMethod.IQAE:
        return iqae_region_estimate(problem, options)

    if method == RegionMeasurementMethod.CLASSICAL_SAMPLING:
        return classical_sampling_region_estimate(
            problem, shots=options.shots, backend=options.backend
        )

    raise ValueError(f"Unsupported region measurement method: {method}")


def classical_sampling_region_estimate(
    problem: RegionEstimationInput,
    *,
    shots: int | None = None,
    backend: object | None = None,
) -> RegionEstimate:
    """
    Estimate the region probability by classical sampling.

    The quantum state is prepared and measured, after which each measured sample
    is classified according to whether it belongs to the selected region.

    Args:
        problem:
            Estimation problem providing the state preparation, sampling
            measurement, and classical classifier.

        shots:
            Number of measurement shots.

        backend:
            Backend used for measurement.

    Returns:
        Estimated region probability and sampling statistics.

    Raises:
        ValueError:
            If the problem does not provide a classical classifier or
            sampling measurement.
    """
    if problem.classical_classifier is None:
        raise ValueError("CLASSICAL_SAMPLING requires classical_classifier.")
    if problem.sampling_measurement is None:
        raise ValueError("CLASSICAL_SAMPLING requires sampling_measurement.")

    args = tuple(problem.qargs())

    problem.state_function(*args)

    counts = problem.sampling_measurement(
        *args,
        shots=shots,
        backend=backend,
    )

    probability = 0.0

    for sample, weight in counts.items():
        if problem.classical_classifier(sample):
            probability += weight

    variance = probability * (1.0 - probability)
    standard_error = None
    if shots is not None and shots > 0:
        standard_error = (variance / shots) ** 0.5
    ci95 = None
    if standard_error is not None:
        ci95 = 1.96 * standard_error

    return RegionEstimate(
        probability=probability,
        method=RegionMeasurementMethod.CLASSICAL_SAMPLING,
        statistics=RegionEstimateStatistics(
            shots=shots,
            sample_variance=variance,
            sample_standard_error=standard_error,
            metadata={
                "confidence_interval_95": ci95,
            },
        ),
    )


def obpa_region_estimate(
    problem: RegionEstimationInput,
    *,
    shots: int | None = None,
    backend: object | None = None,
) -> RegionEstimate:
    """
    Estimate the region probability by directly measuring a flag qubit.

    The state is prepared, the region flag oracle is applied, and the target
    qubit is measured. Its marked-state probability gives the estimated
    probability mass inside the region.

    Args:
        problem: Region-estimation problem providing a flag oracle.
        shots: Number of measurement shots.
        backend: Backend used for measurement.

    Returns:
        Direct sampling estimate and associated sampling statistics.

    Raises:
        ValueError: If no flag oracle is provided.
    """

    if problem.flag_oracle is None:
        raise ValueError("OBPA requires a flag oracle.")

    args = tuple(problem.resolved_qargs())
    target = args[-1]

    problem.state_function(*args)
    problem.flag_oracle(*args)

    counts = target.get_measurement(
        shots=shots,
        backend=backend,
    )

    probability = _probability_from_counts(counts)

    variance = probability * (1.0 - probability)
    standard_error = None
    if shots is not None and shots > 0:
        standard_error = (variance / shots) ** 0.5
    ci95 = None
    if standard_error is not None:
        ci95 = 1.96 * standard_error

    return RegionEstimate(
        probability=probability,
        method=RegionMeasurementMethod.OBPA,
        statistics=RegionEstimateStatistics(
            shots=shots,
            sample_variance=variance,
            sample_standard_error=standard_error,
            metadata={
                "confidence_interval_95": ci95,
            },
        ),
    )


def iqae_region_estimate(
    problem: RegionEstimationInput,
    options: EstimationOptions,
) -> RegionEstimate:
    """
    Estimate the region probability using Iterative Quantum Amplitude Estimation.

    The supplied flag oracle is incorporated into the state function expected by
    the Qrisp IQAE implementation.

    Args:
        problem:
            Region-estimation problem providing the state preparation and
            flag oracle.

        options:
            Runtime configuration containing the requested accuracy, failure
            probability, and measurement settings.

    Returns:
        Estimated region probability together with the requested error bounds
        and estimation statistics.

    Raises:
        ValueError:
            If no flag oracle is provided.
    """

    if problem.flag_oracle is None:
        raise ValueError("IQAE requires a flag oracle.")

    from qrisp import IQAE

    mes_kwargs = _measurement_kwargs(options)

    def state_with_flag(*args: Any) -> None:
        """Prepare the state and flag the portion belonging to the target region."""
        problem.state_function(*args)
        problem.nonnone_flag_oracle(*args)

    probability = float(
        IQAE(
            problem.resolved_qargs,
            state_with_flag,
            eps=options.eps,
            alpha=options.alpha,
            mes_kwargs=mes_kwargs,
        )
    )

    return RegionEstimate(
        probability=probability,
        lower_bound=max(0.0, probability - options.eps),
        upper_bound=min(1.0, probability + options.eps),
        method=RegionMeasurementMethod.IQAE,
        statistics=RegionEstimateStatistics(
            requested_eps=options.eps,
            requested_alpha=options.alpha,
            confidence_level=1.0 - options.alpha,
            theoretical_error_bound=options.eps,
            shots=None,
        ),
    )


def qae_region_estimate(
    problem: RegionEstimationInput,
    options: EstimationOptions,
) -> RegionEstimate:
    """
    Estimate the region probability using canonical Quantum Amplitude Estimation.

    A supplied phase oracle is used directly. If only a flag oracle is available,
    it is converted into the required phase-oracle operation using a Z gate on
    the target qubit.

    Args:
        problem:
            Region-estimation problem providing a phase or flag oracle.

        options:
            Runtime configuration containing the QAE precision, optional target
            register, and measurement settings.

    Returns:
        Estimated region probability and QAE statistics.

    Raises:
        ValueError:
            If neither a phase oracle nor a flag oracle is provided.
    """

    from qrisp import QAE, z

    args = tuple(problem.resolved_qargs())

    if problem.phase_oracle is not None:
        oracle_function = problem.phase_oracle
    elif problem.flag_oracle is not None:

        def oracle_function(*oracle_args: Any) -> None:
            """Convert the flag oracle into a phase oracle using the target qubit."""
            target = oracle_args[-1]

            problem.nonnone_flag_oracle(*oracle_args)
            z(target)
            problem.nonnone_flag_oracle(*oracle_args)

    else:
        raise ValueError("QAE requires either a phase oracle or a flag oracle.")

    angle = QAE(
        args=args,
        state_function=problem.state_function,
        oracle_function=oracle_function,
        precision=options.resolved_precision,
        target=options.qae_target,
    )

    probability = _qae_angle_to_probability(
        angle,
        shots=options.shots,
        backend=options.backend,
    )

    return RegionEstimate(
        probability=probability,
        method=RegionMeasurementMethod.QAE,
        statistics=RegionEstimateStatistics(
            qae_precision=options.resolved_precision,
            theoretical_error_bound=1.0 / (2**options.resolved_precision),
            shots=options.shots,
        ),
    )


def _probability_from_counts(
    counts: MeasurementDict,
) -> float:
    """
    Extract the marked-state probability from flag measurement results.

    Truth-like outcomes representing the marked state are accumulated into
    the returned probability.

    Args:
        counts: Mapping from measurement outcomes to probability weights.

    Returns:
        Total probability of the marked state.
    """

    truth_values = {
        True,
        "1",
        "True",
        "true",
    }

    return float(
        sum(
            probability
            for outcome, probability in counts.items()
            if outcome in truth_values
        )
    )


def _qae_angle_to_probability(
    angle: Any,
    *,
    shots: int | None = None,
    backend: object | None = None,
) -> float:
    """
    Convert a QAE phase estimate into a probability.

    QAE encodes the desired probability as

        a = sin²(pi * theta),

    where `theta` is the estimated phase. If `angle` is a quantum object,
    its measurement distribution is converted to the corresponding weighted
    probability.

    Args:
        angle: Classical or quantum phase estimate returned by QAE.
        shots: Number of shots used when measuring a quantum result.
        backend: Backend used for measurement.

    Returns:
        Estimated probability.

    Raises:
        TypeError: If the QAE result cannot be converted to a probability.
    """

    if isinstance(angle, (float, int)):
        return float(sin(pi * float(angle)) ** 2)

    if hasattr(angle, "get_measurement"):
        counts = angle.get_measurement(
            shots=shots,
            backend=backend,
        )

        probability = 0.0

        for theta, weight in counts.items():
            probability += float(weight) * sin(pi * float(theta)) ** 2

        return probability

    raise TypeError(f"Cannot convert QAE result of type {type(angle)}")


def _measurement_kwargs(
    options: EstimationOptions,
) -> dict[str, Any]:
    """
    Construct the measurement keyword arguments passed to Qrisp.

    Explicit `backend` and `shots` values from the estimation options are
    added to a copy of `mes_kwargs`.

    Args:
        options: Estimation options containing the measurement configuration.

    Returns:
        Keyword arguments for Qrisp measurement routines.
    """

    kwargs = dict(options.mes_kwargs)

    if options.backend is not None:
        kwargs["backend"] = options.backend

    if options.shots is not None:
        kwargs["shots"] = options.shots

    return kwargs
