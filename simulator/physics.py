"""Deterministic phase-1 educational physics, independent of Streamlit.

All model defaults are hypothetical. ``ei`` uses the IEC calibration scaling
as a teaching surrogate; it does not reproduce a calibrated device's EI.
Air kerma here is at the detector, excluding backscatter, not patient dose.
"""

from dataclasses import dataclass
import math
from numbers import Real


def _finite(name: str, value: float, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")


@dataclass(frozen=True)
class ExposureSettings:
    kvp: float = 80.0
    mas: float = 5.0
    sid_cm: float = 100.0
    thickness_cm: float = 15.0

    def __post_init__(self) -> None:
        _finite("kvp", self.kvp)
        _finite("mas", self.mas)
        _finite("sid_cm", self.sid_cm)
        _finite("thickness_cm", self.thickness_cm, allow_zero=True)
        if self.thickness_cm >= self.sid_cm:
            raise ValueError("thickness_cm must be smaller than sid_cm")


@dataclass(frozen=True)
class ModelParameters:
    # Hypothetical open-beam output at the reference detector distance.
    output_ugy_per_mas_at_reference: float = 50.0
    reference_kvp: float = 80.0
    reference_sid_cm: float = 100.0
    kvp_exponent: float = 2.0
    mu_ref_per_cm: float = 0.18
    mu_kvp_exponent: float = 1.0
    grid_transmission: float = 1.0
    target_ei: float = 250.0
    # Effective detected quanta per pixel per microgray, not a measured DQE.
    quanta_per_ugy: float = 1000.0

    def __post_init__(self) -> None:
        for name in (
            "output_ugy_per_mas_at_reference", "reference_kvp",
            "reference_sid_cm", "kvp_exponent",
            "grid_transmission", "target_ei", "quanta_per_ugy",
        ):
            _finite(name, getattr(self, name))
        _finite("mu_ref_per_cm", self.mu_ref_per_cm, allow_zero=True)
        _finite("mu_kvp_exponent", self.mu_kvp_exponent, allow_zero=True)
        if self.grid_transmission > 1:
            raise ValueError("grid_transmission must be at most 1")


@dataclass(frozen=True)
class ExposureResult:
    detector_kerma_ugy: float
    ei: float
    di: float
    effective_mu_per_cm: float
    transmission: float
    mean_quanta: float
    relative_noise: float
    snr: float


def exposure_index(kerma_ugy: float) -> float:
    """IEC calibration scaling EI = 100 μGy⁻¹ × K (dimensionless).

Outside the specified calibration conditions this is a model, not an
equipment-specific inverse calibration of image-region pixel values.
    """
    _finite("kerma_ugy", kerma_ugy)
    ei = 100.0 * kerma_ugy
    _finite("calculated EI", ei)
    return ei


def deviation_index(ei: float, target_ei: float) -> float:
    """DI = 10 log10(EI / EI_T), with no clinical action thresholds."""
    _finite("ei", ei)
    _finite("target_ei", target_ei)
    # Difference of logs avoids overflow/underflow of the intermediate ratio.
    return 10.0 * (math.log10(ei) - math.log10(target_ei))


def calculate_exposure(
    settings: ExposureSettings,
    parameters: ModelParameters = ModelParameters(),
) -> ExposureResult:
    """Apply inverse-square, empirical kVp scaling and effective attenuation.

    K = output × mAs × (kVp/kVp_ref)^n × (SID_ref/SID)^2
        × exp(-mu × thickness) × grid_transmission
    mu = mu_ref × (kVp_ref/kVp)^p
    N = K × quanta_per_ugy; relative_noise = 1/sqrt(N); SNR = sqrt(N).

    Numeric overflow/underflow is reported explicitly, never silently clamped.
    UI ranges are a narrower pedagogic domain than this calculation API.
    """
    if not isinstance(settings, ExposureSettings):
        raise TypeError("settings must be ExposureSettings")
    if not isinstance(parameters, ModelParameters):
        raise TypeError("parameters must be ModelParameters")
    try:
        log_kvp_ratio = math.log(settings.kvp) - math.log(parameters.reference_kvp)
        mu = (
            parameters.mu_ref_per_cm * math.exp(-parameters.mu_kvp_exponent * log_kvp_ratio)
            if parameters.mu_ref_per_cm else 0.0
        )
        transmission = math.exp(-mu * settings.thickness_cm)
        log_kerma = (
            math.log(parameters.output_ugy_per_mas_at_reference)
            + math.log(settings.mas)
            + parameters.kvp_exponent * log_kvp_ratio
            + 2 * (math.log(parameters.reference_sid_cm) - math.log(settings.sid_cm))
            - mu * settings.thickness_cm
            + math.log(parameters.grid_transmission)
        )
        kerma = math.exp(log_kerma)
        mean_quanta = kerma * parameters.quanta_per_ugy
        _finite("calculated attenuation", mu, allow_zero=True)
        _finite("calculated transmission", transmission)
        _finite("calculated detector kerma", kerma)
        _finite("calculated mean quanta", mean_quanta)
        ei = exposure_index(kerma)
        snr = math.sqrt(mean_quanta)
        return ExposureResult(
            detector_kerma_ugy=kerma,
            ei=ei,
            di=deviation_index(ei, parameters.target_ei),
            effective_mu_per_cm=mu,
            transmission=transmission,
            mean_quanta=mean_quanta,
            relative_noise=1.0 / snr,
            snr=snr,
        )
    except OverflowError as error:
        raise ValueError("Model calculation exceeded the supported numeric range") from error
