class ADVVError(Exception):
    """An actionable pipeline error."""


class ConfigError(ADVVError):
    pass


class DataError(ADVVError):
    pass


class ResponseError(ADVVError):
    pass


class BackendError(ADVVError):
    """An image-specific failure; the configured retry budget applies."""


class FatalBackendError(BackendError):
    """Loading, resource, or environment failure; stop and preserve the run."""


class RunCancelled(ADVVError):
    pass


class SamplingSkipped(DataError):
    """No valid geometry for one object_region_v2 attempt; the attempt is skipped, not failed."""
