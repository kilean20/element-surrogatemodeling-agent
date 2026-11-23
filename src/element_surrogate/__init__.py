"""
element_surrogate - tools for building neural-network surrogates of one magnet's
particle transfer map from its raw 3D field map.

Every tool is a plain function that works without an agent. The MCP server
(`element_surrogate.mcp_server`) serves the same tools to agent clients and writes a
run record for every call (`element_surrogate.runs`).
"""

__version__ = "0.2.0rc29"
__license__ = "Apache-2.0"

__all__ = [
    "__version__",
    "__license__",
]
