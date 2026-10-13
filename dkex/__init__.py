"""Dynamic Kidney Exchange (dkex) — module-wise research pipeline.

Implements the framework of the concept note "Dynamic Kidney Exchange
Optimisation Under Realistic Clinical Constraints for Equitable Allocation".

Module map (concept-note section -> module):

    3.1  Dynamic pool model ............ dkex.simulation.dynamics
    3.2  Compatibility model ........... dkex.simulation.compatibility
    3.3  Logistics feasibility ......... dkex.simulation.logistics
    4.1  Synthetic pool generation ..... dkex.simulation.generator
         Cycle / chain enumeration ..... dkex.optimisation.structures
    3.4  Fairness objectives F1-F4 ..... dkex.optimisation.objectives
    4.2  NSGA-III (and NSGA-II) ........ dkex.optimisation.moea
    3.5  Adaptive operator selection ... dkex.aos
    4.4  ILP / lexicographic baselines . dkex.optimisation.ilp
    3.6  Price of Fairness ............. dkex.analysis.pof
    3.7  Clearing strategies ........... dkex.simulation.clearing
    4.4  Baseline configurations ....... dkex.experiments
    4.5  Metrics & statistics .......... dkex.analysis.metrics / dkex.analysis.stats
    4.6  PrefLib OOD validation ........ dkex.data.preflib
"""

__version__ = "0.1.0"
