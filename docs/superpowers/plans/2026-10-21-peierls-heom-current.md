# Peierls HEOM-TT implementation plan

1. Add failing dense-algebra tests for generic left system and bath-coordinate
   actions, then implement and export the two APIs.
2. Add the periodic Peierls model, bath decomposition, and extended current
   builder. Test the three-site ring model and nonzero-ADO current action.
3. Add equilibrium and CCF entry points, checkpoint metadata, CSV, plots, and
   a concise README, following the Holstein workflow.
4. Add an independent finite-oscillator benchmark for a three-site ring and
   run focused tests plus a small example smoke test.
