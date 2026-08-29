"""Dota 2 Meta Finder -- empirical meta analysis from public match data.

Answers the draft-and-macro questions (tier list, counters, synergy, comp
scoring, next-pick recommendation) by aggregating OpenDota public-match data.
Read-only official API; no automated play.

The reinforcement-learning sandbox from the original spec lives on the
``archive/rl-sandbox`` branch / ``rl-sandbox-v1`` tag.
"""
__version__ = "0.1.0"

from . import counters, draft, heroes, opendota, store, synergy  # noqa: F401

__all__ = ["opendota", "heroes", "store", "counters", "synergy", "draft"]
