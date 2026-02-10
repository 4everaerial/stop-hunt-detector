"""
Context Module - Data Products for Situational Awareness

This module provides data alignment and snapshot capabilities
for market context streams (open interest, funding rates, on-chain).

These are CONTEXT STREAMS only. They do NOT:
- Modify stress scores
- Influence detector logic
- Provide predictions or trading signals

Purpose: Situational awareness for risk management.
"""

from .snapshot import ContextSnapshot, create_snapshot

__all__ = ['ContextSnapshot', 'create_snapshot']
