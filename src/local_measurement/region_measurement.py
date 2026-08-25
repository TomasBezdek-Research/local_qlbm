"""
Problem-independent region measurement utilities.

This module is intentionally independent of any specific application
(e.g. QLBM). The user provides:

1. A register factory (`qargs`)
2. A state preparation / simulation function (`state_function`)
3. Either:
    - a flag oracle
    - a phase oracle

A region measurement method can then be selected through
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


class RegionMeasurementMethod(str, Enum):
    """
    Supported region estimation methods.
    """

    OBPA = "obpa"
    QAE = "qae"
    IQAE = "iqae"
    CLASSICAL_SAMPLING = "classical_sampling"


@dataclass(frozen=True)
class RegionEstimate:
    """
    Result returned by a region estimation algorithm.

    Args:
        probability:
            Estimated probability mass in the region.

        lower_bound:
            Optional lower confidence bound.

        upper_bound:
            Optional upper confidence bound.

        method:
            Method used to generate the estimate.
    """

    probability: float

    lower_bound: float | None = None
    upper_bound: float | None = None

    method: RegionMeasurementMethod | None = None

    statistics: RegionEstimateStatistics | None = None


@dataclass(frozen=True)
class RegionEstimateStatistics:
    """
    Auxiliary information about the estimation procedure.
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
    Generic circuit description for region-density estimation.

    Args:
        qargs:
            Factory returning all quantum registers required by
            the algorithm.

        state_function:
            Function preparing the quantum state whose region
            probability shall be estimated.
            Take all the quantum registers returned by qargs()
            as arguments, and if necessary for the method, also target
            qubit.

        flag_oracle:
            Oracle marking good states into a target qubit.

        phase_oracle:
            Oracle applying a phase flip to good states.

        classical_classifier:
            Optional classifier function for classical sampling.

        target_factory:
            Optional factory creating a fresh target qubit.
            This is only required when the target qubit is not
            already part of the registers returned by qargs().

        metadata:
            Additional user-defined information associated
            with the estimation problem.
    """

    qargs: QArgsFactory
    state_function: StateFunction

    flag_oracle: FlagOracle | None = None

    @property
    def nonnone_flag_oracle(self) -> FlagOracle:
        if self.flag_oracle is None:
            raise ValueError("flag_oracle is None")
        return self.flag_oracle

    phase_oracle: PhaseOracle | None = None

    @property
    def nonnone_phase_oracle(self) -> PhaseOracle:
        if self.phase_oracle is None:
            raise ValueError("phase_oracle is None")
        return self.phase_oracle

    classical_classifier: Callable[[Any], bool] | None = None

    target: TargetFactory | None = None

    @property
    def nonnone_target(self) -> TargetFactory:
        if self.target is None:
            raise ValueError("target is None")
        return self.target

    sampling_measurement: Callable[..., Any] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_qargs(self) -> QArgsFactory:
        """
        Returns a qargs factory with a unified convention:

        The returned qargs always contain the target qubit
        as the last argument.

        Returns:
            QArgsFactory:
                Factory function that returns the qargs tuple.
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
    Runtime configuration for region estimation.

    Args:
        method:
            Region estimation method.

        eps:
            Target estimation accuracy.

        alpha:
            Failure probability.

        qae_precision:
            Precision used for canonical QAE.
            If None, it is computed from `eps`.

        shots:
            Number of measurement shots.

        backend:
            Execution backend.

        mes_kwargs:
            Additional measurement arguments passed to Qrisp.

        qae_target:
            Target quantum float for QAE, into which the estimation is perform.
            If both `qae_target` and `qae_precision` are None, `qae_precision` is computed from `eps`.
            If `qae_target` is not None, `qae_precision` is ignored.

    """

    method: RegionMeasurementMethod | None = None

    eps: float = 0.01
    alpha: float = 0.05

    qae_precision: int | None = None

    @property
    def resolved_precision(self) -> int:
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
    Estimate the probability mass contained in the selected region.

    Dispatches the estimation task to the selected backend.

    Defaults to `RegionMeasurementMethod.IQAE` if `estimation_method` is None and method is not specified in `options`.

    Args:
        problem:
            Region estimation problem definition containing the
            state preparation and oracle descriptions.

        options:
            Runtime configuration of the estimation algorithm.
            If None, default options are used.

        estimation_method:
            Override the estimation method to use.
            If None, the method is determined from `options`.

    Returns:
        RegionEstimate:
            Estimated region probability together with optional
            confidence bounds.
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

    raise ValueError(f"Unsupported region measurement method: {options.method}")


def classical_sampling_region_estimate(
    problem: RegionEstimationInput,
    *,
    shots: int | None = None,
    backend: object | None = None,
) -> RegionEstimate:
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
    Estimate the region probability through direct flag measurement.

    The state is prepared once, the flag oracle is applied,
    and the target qubit is measured.

    Args:
        problem:
            Region estimation problem definition.

        shots:
            Number of measurement shots.

        backend:
            Backend used for circuit execution.

    Returns:
        RegionEstimate:
            Direct sampling estimate of the region probability.

    Raises:
        ValueError:
            If no flag oracle is provided.
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
    Estimate the region probability using Iterative Quantum
    Amplitude Estimation (IQAE).

    Args:
        problem:
            Region estimation problem definition.

        options:
            IQAE runtime configuration.

            eps:
                Requested estimation accuracy.

            alpha:
                Failure probability.

            shots:
                Measurement shot count.

            backend:
                Backend used for measurements.

    Returns:
        RegionEstimate:
            Estimated region probability and confidence bounds.

    Raises:
        ValueError:
            If no flag oracle is available.
    """

    if problem.flag_oracle is None:
        raise ValueError("IQAE requires a flag oracle.")

    from qrisp import IQAE

    mes_kwargs = _measurement_kwargs(options)

    def state_with_flag(*args: Any) -> None:
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
    Estimate the region probability using canonical Quantum
    Amplitude Estimation (QAE).

    Args:
        problem:
            Region estimation problem definition.

        options:
            QAE runtime configuration.

            precision:
                Estimation precision.

            qae_target:
                Optional target register containing the
                QAE estimate.

    Returns:
        RegionEstimate:
            Estimated region probability.

    Raises:
        ValueError:
            If neither a phase oracle nor a flag oracle
            is available.

        ValueError:
            If neither precision nor qae_target are supplied.
    """

    from qrisp import QAE, z

    args = tuple(problem.resolved_qargs())

    if problem.phase_oracle is not None:
        oracle_function = problem.phase_oracle
    elif problem.flag_oracle is not None:

        def oracle_function(*oracle_args: Any) -> None:
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
    Convert flag measurement results into a probability.

    Args:
        counts:
            Measurement dictionary mapping outcomes
            to probabilities or frequencies.

    Returns:
        float:
            Probability of obtaining the marked state.
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
    Convert a QAE angle estimate into a probability.

    QAE estimates an angle theta satisfying

        a = sin²(pi * theta)

    where a is the desired probability.

    Args:
        angle:
            Result returned by QAE.

        shots:
            Number of measurement shots used when
            measuring a quantum output object.

        backend:
            Backend used for measurement.

    Returns:
        float:
            Estimated probability.

    Raises:
        TypeError:
            If the supplied QAE result cannot be converted.
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
    Create measurement keyword arguments for Qrisp primitives.

    Args:
        options:
            Estimation options containing measurement
            configuration.

    Returns:
        dict[str, Any]:
            Dictionary of keyword arguments passed
            to Qrisp measurement routines.
    """

    kwargs = dict(options.mes_kwargs)

    if options.backend is not None:
        kwargs["backend"] = options.backend

    if options.shots is not None:
        kwargs["shots"] = options.shots

    return kwargs
