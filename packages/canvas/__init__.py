"""CANVAS — Coordinating Austere Nodes through Virtualization and Analysis of Streams.

Heli.OS implementation of AFRL/RIK BAA FA8750-24-S-7003 (CANVAS) Technical
Areas 1 (virtual C2 layer) and 2 (decentralized C2 framework).

Modules:
  authority_dsl     — load + validate the OPA-signed authority policy
  workflow_sim      — virtual TA1 simulator that runs proposed business-rule
                      changes against a synthetic operational scenario and
                      surfaces the propagation graph
  intent_push       — sign + push intent + policy bundles to TA2 nodes
"""
