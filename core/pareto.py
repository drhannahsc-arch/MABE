"""
core/pareto.py — Multi-Objective Pareto Optimization for De Novo Design

NSGA-II style non-dominated sorting with crowding distance.
Operates on objective vectors extracted from GeneratedCandidate objects.

Three default objectives (all MAXIMIZED internally):
  1. Affinity:     log_Ka_pred (higher = stronger binding)
  2. Selectivity:  min_gap (higher = more selective over interferents)
  3. Synthesizability: -sa_score_val (lower SA = easier; negated to maximize)

Custom objectives can be defined by passing objective extractor functions.

References:
  Deb et al., IEEE Trans. Evol. Comput. 6(2):182-197, 2002 (NSGA-II)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════
# OBJECTIVE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Objective:
    """Single optimization objective.

    Args:
        name: Display name
        extract: Function that takes a candidate and returns a float
        maximize: True to maximize, False to minimize.
                  Internally, all objectives are converted to maximization.
        weight: Optional weight for weighted-sum fallback (not used in Pareto
                sorting itself, but available for tie-breaking or reporting).
    """
    name: str
    extract: Callable
    maximize: bool = True
    weight: float = 1.0

    def value(self, candidate):
        """Extract raw objective value (before sign flip)."""
        return self.extract(candidate)

    def signed_value(self, candidate):
        """Value for maximization (negated if minimize)."""
        v = self.extract(candidate)
        return v if self.maximize else -v


# Default objectives for de novo selectivity screening
DEFAULT_OBJECTIVES = [
    Objective(
        name="affinity",
        extract=lambda c: c.log_Ka_pred,
        maximize=True,
    ),
    Objective(
        name="selectivity",
        extract=lambda c: c.min_gap,
        maximize=True,
    ),
    Objective(
        name="synthesizability",
        extract=lambda c: c.sa_score_val,
        maximize=False,  # lower SA = better → negated internally
    ),
]

# Objectives for non-selectivity mode (no interferents)
AFFINITY_SA_OBJECTIVES = [
    Objective(
        name="affinity",
        extract=lambda c: c.log_Ka_pred,
        maximize=True,
    ),
    Objective(
        name="synthesizability",
        extract=lambda c: c.sa_score_val,
        maximize=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# PARETO DOMINANCE
# ═══════════════════════════════════════════════════════════════════════════

def dominates(obj_a, obj_b):
    """True if objective vector a dominates b (all maximized).

    a dominates b iff:
      - a[i] >= b[i] for all i, AND
      - a[j] > b[j] for at least one j
    """
    dominated = False
    at_least_one_better = False
    for va, vb in zip(obj_a, obj_b):
        if va < vb:
            return False
        if va > vb:
            at_least_one_better = True
    return at_least_one_better


# ═══════════════════════════════════════════════════════════════════════════
# FAST NON-DOMINATED SORT (NSGA-II Algorithm)
# ═══════════════════════════════════════════════════════════════════════════

def fast_non_dominated_sort(objective_vectors):
    """Assign each individual to a Pareto front.

    Args:
        objective_vectors: list of tuples, each a signed objective vector
                          (all objectives to be maximized).

    Returns:
        List of fronts, where front[0] is the Pareto-optimal set.
        Each front is a list of indices into objective_vectors.
    """
    n = len(objective_vectors)
    if n == 0:
        return []

    # S[p] = set of solutions dominated by p
    S = [[] for _ in range(n)]
    # n_dom[p] = number of solutions that dominate p
    n_dom = [0] * n

    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objective_vectors[p], objective_vectors[q]):
                S[p].append(q)
            elif dominates(objective_vectors[q], objective_vectors[p]):
                n_dom[p] += 1

        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


# ═══════════════════════════════════════════════════════════════════════════
# CROWDING DISTANCE
# ═══════════════════════════════════════════════════════════════════════════

def crowding_distance(front_indices, objective_vectors):
    """Compute crowding distance for members of a single front.

    Measures how isolated each solution is in objective space.
    Higher distance = more isolated = more desirable for diversity.

    Args:
        front_indices: list of indices into objective_vectors
        objective_vectors: full list of signed objective vectors

    Returns:
        dict mapping index → crowding distance (float or float('inf'))
    """
    n = len(front_indices)
    if n <= 2:
        return {idx: float('inf') for idx in front_indices}

    n_obj = len(objective_vectors[front_indices[0]])
    distances = {idx: 0.0 for idx in front_indices}

    for m in range(n_obj):
        # Sort front by objective m
        sorted_front = sorted(front_indices,
                              key=lambda i: objective_vectors[i][m])

        # Boundary points get infinite distance
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        obj_range = (objective_vectors[sorted_front[-1]][m]
                     - objective_vectors[sorted_front[0]][m])
        if obj_range == 0:
            continue

        for k in range(1, n - 1):
            distances[sorted_front[k]] += (
                (objective_vectors[sorted_front[k + 1]][m]
                 - objective_vectors[sorted_front[k - 1]][m])
                / obj_range
            )

    return distances


# ═══════════════════════════════════════════════════════════════════════════
# PARETO RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParetoCandidate:
    """Pareto-annotated candidate."""
    index: int                      # Index into original candidate list
    front: int                      # 0 = Pareto-optimal, 1 = second front, ...
    crowding: float                 # Crowding distance (inf for boundary)
    objectives: dict                # name → raw value (before sign flip)
    pareto_rank: int = 0            # Combined rank (front first, crowding second)


@dataclass
class ParetoResult:
    """Output of Pareto ranking."""
    candidates: list                # ParetoCandidate objects, sorted by pareto_rank
    fronts: list                    # List of fronts (each a list of indices)
    n_pareto_optimal: int           # Size of front 0
    objective_names: list           # Names of objectives used
    hypervolume: float = 0.0        # Optional hypervolume indicator

    @property
    def pareto_front(self):
        """Indices of Pareto-optimal candidates."""
        return self.fronts[0] if self.fronts else []

    def front_candidates(self, front_idx):
        """Get ParetoCandidate objects for a specific front."""
        return [c for c in self.candidates if c.front == front_idx]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def pareto_rank(candidates, objectives=None):
    """Rank candidates by multi-objective Pareto dominance.

    Args:
        candidates: list of GeneratedCandidate (or any object with fields
                    that the objective extractors can read)
        objectives: list of Objective definitions. Defaults to
                    DEFAULT_OBJECTIVES (affinity, selectivity, SA).

    Returns:
        ParetoResult with ranked candidates and front assignments.
    """
    if not candidates:
        return ParetoResult(
            candidates=[], fronts=[], n_pareto_optimal=0,
            objective_names=[]
        )

    if objectives is None:
        # Auto-select: if any candidate has selectivity data, use 3-obj
        if any(getattr(c, 'min_gap', 0.0) != 0.0 for c in candidates):
            objectives = DEFAULT_OBJECTIVES
        else:
            objectives = AFFINITY_SA_OBJECTIVES

    obj_names = [o.name for o in objectives]

    # Extract signed objective vectors
    obj_vectors = []
    for c in candidates:
        vec = tuple(o.signed_value(c) for o in objectives)
        obj_vectors.append(vec)

    # Non-dominated sort
    fronts = fast_non_dominated_sort(obj_vectors)

    # Crowding distance per front
    all_crowding = {}
    for front in fronts:
        cd = crowding_distance(front, obj_vectors)
        all_crowding.update(cd)

    # Build ParetoCandidate objects
    pareto_candidates = []
    for front_idx, front in enumerate(fronts):
        for idx in front:
            raw_obj = {o.name: o.value(candidates[idx]) for o in objectives}
            pc = ParetoCandidate(
                index=idx,
                front=front_idx,
                crowding=all_crowding.get(idx, 0.0),
                objectives=raw_obj,
            )
            pareto_candidates.append(pc)

    # Sort by (front ASC, crowding DESC) — NSGA-II crowded comparison
    pareto_candidates.sort(key=lambda p: (p.front, -p.crowding))
    for rank, pc in enumerate(pareto_candidates):
        pc.pareto_rank = rank + 1

    # Hypervolume (2D projection: affinity × SA for quick indicator)
    hv = _hypervolume_2d(obj_vectors, fronts[0]) if fronts else 0.0

    return ParetoResult(
        candidates=pareto_candidates,
        fronts=fronts,
        n_pareto_optimal=len(fronts[0]) if fronts else 0,
        objective_names=obj_names,
        hypervolume=hv,
    )


def _hypervolume_2d(obj_vectors, front_indices):
    """Approximate hypervolume using first two objectives.

    Reference point: (min_obj1 - 1, min_obj2 - 1).
    Only a rough indicator; exact n-D hypervolume is O(n^(d-1) log n).
    """
    if not front_indices or len(obj_vectors[0]) < 2:
        return 0.0

    points = [(obj_vectors[i][0], obj_vectors[i][1]) for i in front_indices]
    ref_x = min(p[0] for p in points) - 1.0
    ref_y = min(p[1] for p in points) - 1.0

    # Sort by x descending
    points.sort(key=lambda p: -p[0])

    hv = 0.0
    prev_y = ref_y
    for x, y in points:
        if y > prev_y:
            hv += (x - ref_x) * (y - prev_y)
            prev_y = y

    return hv


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_pareto(result, candidates, top_n=20):
    """Print Pareto ranking results.

    Args:
        result: ParetoResult
        candidates: original candidate list (for SMILES, name, etc.)
    """
    print(f"\n{'='*78}")
    print(f"PARETO RANKING — {len(result.candidates)} candidates, "
          f"{len(result.fronts)} fronts")
    print(f"Objectives: {', '.join(result.objective_names)}")
    print(f"Pareto-optimal (front 0): {result.n_pareto_optimal} candidates")
    if result.hypervolume > 0:
        print(f"Hypervolume (2D approx): {result.hypervolume:.2f}")
    print(f"{'='*78}")

    header = (f"{'Rank':>4} {'Front':>5} {'Crowd':>7} "
              + " ".join(f"{name:>12}" for name in result.objective_names)
              + f"  {'Name'}")
    print(header)
    print("-" * len(header))

    for pc in result.candidates[:top_n]:
        c = candidates[pc.index]
        crowd_str = ("  inf" if pc.crowding == float('inf')
                     else f"{pc.crowding:7.3f}")
        obj_str = " ".join(
            f"{pc.objectives[name]:12.3f}"
            for name in result.objective_names
        )
        name = getattr(c, 'name', getattr(c, 'smiles', str(pc.index))[:30])
        front_marker = " *" if pc.front == 0 else "  "
        print(f"{pc.pareto_rank:4d} {pc.front:5d}{front_marker}{crowd_str} "
              f"{obj_str}  {name}")

    if len(result.candidates) > top_n:
        print(f"  ... ({len(result.candidates) - top_n} more)")


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick smoke test with synthetic data
    @dataclass
    class FakeCandidate:
        log_Ka_pred: float = 0.0
        min_gap: float = 0.0
        sa_score_val: float = 5.0

    # 3 candidates on the Pareto front, 2 dominated
    fakes = [
        FakeCandidate(log_Ka_pred=20.0, min_gap=5.0, sa_score_val=3.0),  # high aff, good SA
        FakeCandidate(log_Ka_pred=15.0, min_gap=10.0, sa_score_val=4.0), # high sel
        FakeCandidate(log_Ka_pred=18.0, min_gap=8.0, sa_score_val=2.0),  # easy synth
        FakeCandidate(log_Ka_pred=14.0, min_gap=4.0, sa_score_val=6.0),  # dominated
        FakeCandidate(log_Ka_pred=12.0, min_gap=3.0, sa_score_val=7.0),  # dominated
    ]
    result = pareto_rank(fakes)
    print_pareto(result, fakes)
    assert result.n_pareto_optimal == 3, f"Expected 3 on front 0, got {result.n_pareto_optimal}"
    print("\nSelf-test PASSED")
