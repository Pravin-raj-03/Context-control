"""Shared, dependency-free primitives: message/result dataclasses, the
component registry, and the validated configuration schema. Kept isolated so
that ``policies``, ``chatbot``, ``engine``, and ``evaluation`` never import one
another transitively through shared types.
"""
