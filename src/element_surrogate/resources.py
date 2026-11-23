"""Explicit numerical thread counts shared by managed scientific backends."""

import os


def numerical_threads(requested):
    """Validate a backend's declared count against its cooperating job reservation."""
    if type(requested) is not int or requested < 1:
        raise ValueError("positive integer numerical thread count required")
    maximum = os.environ.get("ELEMENT_SURROGATE_NUMERICAL_THREADS")
    if maximum is not None and requested > int(maximum):
        raise ValueError("backend numerical threads exceed the managed job reservation")
    return requested


def configure_torch_threads(torch, requested):
    """Set a supplied count; managed workers start with sequential inter-op work."""
    count = numerical_threads(requested)
    if os.environ.get("ELEMENT_SURROGATE_NUMERICAL_THREADS") is not None:
        if torch.get_num_interop_threads() != 1:
            # PyTorch permits this only before inter-op work starts. A late
            # invalid reconfiguration must fail visibly, never oversubscribe.
            torch.set_num_interop_threads(1)
    torch.set_num_threads(count)
