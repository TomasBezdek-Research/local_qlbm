"""
Analysis utilities for region-probability estimation.

This module mirrors the region-estimation methods defined in
`region_measurement`, but additionally returns the quantum sessions and
measurement-shot counts generated during each estimation.

These additional results can be used to analyze circuit resources such as
qubit count, circuit depth, gate count, and measurement cost.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import pi, sin
from typing import Any

from qrisp import QuantumBool, QuantumSession

from .region_measurement import (
    EstimationOptions,
    RegionEstimate,
    RegionEstimateStatistics,
    RegionEstimationInput,
    RegionMeasurementMethod,
)

QArgs = Sequence[Any]
QArgsFactory = Callable[[], QArgs]
TargetFactory = Callable[[], QuantumBool]

StateFunction = Callable[..., None]
FlagOracle = Callable[..., None]
PhaseOracle = Callable[..., None]

MeasurementDict = Mapping[Any, float]


def estimate_region_probability_analysis(
    problem: RegionEstimationInput,
    options: EstimationOptions | None = None,
    estimation_method: RegionMeasurementMethod | None = None,
) -> tuple[
    RegionEstimate,
    QuantumSession | Sequence[QuantumSession],
    int | None | Sequence[int | None],
]:
    """
    Estimate a region probability and return circuit-analysis information.

    The estimation method is selected explicitly through `estimation_method`,
    through `options.method`, or defaults to IQAE. In addition to the region
    estimate, the function returns the quantum session or sessions generated
    during the estimation and their corresponding measurement-shot counts.

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
        Region estimate, generated quantum session or sessions, and corresponding
        measurement-shot count or counts.

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
) -> tuple[
    RegionEstimate,
    QuantumSession | Sequence[QuantumSession],
    int | None | Sequence[int | None],
]:
    """
    Estimate the region probability by classical sampling.

    The state is prepared and measured, and each sample is classified according
    to whether it belongs to the selected region. The quantum session used for
    the measurement and the configured shot count are returned for analysis.

    Args:
        problem:
            Estimation problem providing the state preparation, sampling
            measurement, and classical classifier.

        shots:
            Number of measurement shots.

        backend:
            Backend used for measurement.

    Returns:
        Region estimate, quantum session used for sampling, and shot count.

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

    return (
        RegionEstimate(
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
        ),
        args[0].qs,
        shots,
    )


def obpa_region_estimate(
    problem: RegionEstimationInput,
    *,
    shots: int | None = None,
    backend: object | None = None,
) -> tuple[
    RegionEstimate,
    QuantumSession | Sequence[QuantumSession],
    int | None | Sequence[int | None],
]:
    """
    Estimate the region probability by directly measuring a flag qubit.

    The state is prepared, the region flag oracle is applied, and the target
    qubit is measured. The resulting session and measurement-shot count are
    returned together with the probability estimate for circuit analysis.

    Args:
        problem:
            Region-estimation problem providing a flag oracle.

        shots:
            Number of measurement shots.

        backend:
            Backend used for measurement.

    Returns:
        Region estimate, quantum session used for the flag measurement,
        and shot count.

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

    return (
        RegionEstimate(
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
        ),
        target.qs,
        shots,
    )


def iqae_region_estimate(
    problem: RegionEstimationInput,
    options: EstimationOptions,
) -> tuple[
    RegionEstimate,
    QuantumSession | Sequence[QuantumSession],
    int | None | Sequence[int | None],
]:
    """
    Estimate the region probability using Iterative Quantum Amplitude Estimation.

    The flag oracle is incorporated into the state-preparation function expected
    by the IQAE analysis implementation. All quantum sessions and corresponding
    shot counts generated during the iterative procedure are returned for
    resource analysis.

    Args:
        problem:
            Region-estimation problem providing the state preparation and
            flag oracle.

        options:
            IQAE runtime configuration containing the requested accuracy,
            failure probability, and measurement settings.

    Returns:
        Region estimate, quantum sessions generated by IQAE, and the
        corresponding measurement-shot counts.

    Raises:
        ValueError:
            If no flag oracle is provided.
    """

    if problem.flag_oracle is None:
        raise ValueError("IQAE requires a flag oracle.")

    from .iterative_qae_analysis import IQAE as IQAE_analysis

    mes_kwargs = _measurement_kwargs(options)

    def state_with_flag(*args: Any) -> None:
        """Prepare the state and flag states belonging to the target region."""
        problem.state_function(*args)
        problem.nonnone_flag_oracle(*args)

    probability, quantum_sessions, shots = IQAE_analysis(
        problem.resolved_qargs,
        state_with_flag,
        eps=options.eps,
        alpha=options.alpha,
        mes_kwargs=mes_kwargs,
    )
    probability = float(probability)

    return (
        RegionEstimate(
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
        ),
        quantum_sessions,
        shots,
    )


def qae_region_estimate(
    problem: RegionEstimationInput,
    options: EstimationOptions,
) -> tuple[
    RegionEstimate,
    QuantumSession | Sequence[QuantumSession],
    int | None | Sequence[int | None],
]:
    """
    Estimate the region probability using canonical Quantum Amplitude Estimation.

    A supplied phase oracle is used directly. If only a flag oracle is available,
    it is converted into a phase oracle using a Z gate on the target qubit.

    The quantum session and configured measurement-shot count are returned
    together with the estimate for circuit analysis.

    Args:
        problem:
            Region-estimation problem providing a phase or flag oracle.

        options:
            Runtime configuration containing the QAE precision, optional target
            register, and measurement settings.

    Returns:
        Region estimate, quantum session used by QAE, and measurement-shot count.

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
            """Convert the flag oracle into a pha*e oracle using the target qubit."""
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

    return (
        RegionEstimate(
            probability=probability,
            method=RegionMeasurementMethod.QAE,
            statistics=RegionEstimateStatistics(
                qae_precision=options.resolved_precision,
                theoretical_error_bound=1.0 / (2**options.resolved_precision),
                shots=options.shots,
            ),
        ),
        args[0].qs,
        options.shots,
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
    its measurement distribution is converted into the corresponding
    weighted probability.

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
    Construct measurement keyword arguments passed to Qrisp.

    A copy of `mes_kwargs` is extended with the configured `backend` and `shots`.
    Explicit `backend` and `shots` values override corresponding entries already
    present in `mes_kwargs`.

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
