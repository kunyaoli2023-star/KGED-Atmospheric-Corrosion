"""Evaluated-search stopping, independent of model prose or proposal counts.

Fresh novelty is not an improvement. All admitted proposals must first receive
their declared numerical evaluation. Solver exclusions are kept distinct from
mathematical rejection and from complete finite-library verification.
"""
from __future__ import annotations

PROTOCOL_ID = 'evaluated_round_pareto_semantic_stability_20261008'
EPSILON = 0.002


def stable_transition(prev_front, cur_front, eps: float = EPSILON) -> bool:
    """Two-round Pareto-front stability criterion (Main Manuscript Eq. (8), eps = 0.002).

    Every member p of the current Pareto front P_r must be matched by at least one
    member q of the preceding front P_{r-1} satisfying:
        E_q <= E_p + eps,  k_q <= k_p,  s_q <= s_p.
    """
    if not prev_front or not cur_front:
        return False
    for p in cur_front:
        sp = p.get('score', p)
        e_p = float(sp['cv_log_rmse'])
        k_p = int(sp['parameter_count'])
        s_p = int(sp.get('canonical_ast_nodes', sp.get('ast_nodes')))
        covered = False
        for q in prev_front:
            sq = q.get('score', q)
            s_q = int(sq.get('canonical_ast_nodes', sq.get('ast_nodes')))
            if (
                float(sq['cv_log_rmse']) <= e_p + eps + 1e-12
                and int(sq['parameter_count']) <= k_p
                and s_q <= s_p
            ):
                covered = True
                break
        if not covered:
            return False
    return True


def scientific_change_or_pending(parent_changed, new_contrasts, uncertain_contrasts, pending_parent_updates):
    """Unresolved scientific adjudication blocks stability, not a claim of novelty."""
    return bool(parent_changed or new_contrasts or uncertain_contrasts or pending_parent_updates)


def assess(*, previous_stable_rounds, material_progress, hypothesis_changed,
           evaluation_complete, pending_candidates, numerical_exclusions,
           patience, scope='finite_library', post_selection_review_pending=False):
    if scope not in ('finite_library', 'evaluated_round_candidates'):
        raise ValueError('Unknown stopping scope')
    if patience < 2:
        raise ValueError('At least two complete challenge cycles are required')
    complete = bool(evaluation_complete) and pending_candidates == 0
    stable = complete and not material_progress and not hypothesis_changed and not post_selection_review_pending
    streak = previous_stable_rounds + 1 if stable else 0
    stop = streak >= patience
    blockers = []
    if not complete:
        blockers.append('EVALUATION_INCOMPLETE')
    if material_progress:
        blockers.append('PARETO_IMPROVED')
    if hypothesis_changed:
        blockers.append('HYPOTHESIS_CHANGED_OR_PENDING')
    if post_selection_review_pending:
        blockers.append('SELECTED_EQUATION_REVIEW_PENDING')
    if stable and not stop:
        blockers.append('STABILITY_CONFIRMATION_PENDING')
    if stop and scope == 'evaluated_round_candidates':
        reason = ('EVALUATED_ROUND_PARETO_STABLE_WITH_NUMERICAL_EXCLUSIONS'
                  if numerical_exclusions else 'EVALUATED_ROUND_PARETO_STABLE')
    elif stop and numerical_exclusions:
        reason = 'OPERATIONAL_PARETO_STABLE_WITH_NUMERICAL_EXCLUSIONS'
    elif stop:
        reason = 'DECLARED_FINITE_LIBRARY_STABLE'
    else:
        reason = 'CONTINUE'
    return {
        'convergence_protocol': PROTOCOL_ID,
        'stability_scope': scope,
        'stable_rounds': streak,
        'required_stable_rounds': patience,
        'continuation_reasons': blockers,
        'stop': stop,
        'stop_reason': reason,
        'operational_search_stable': stop,
        'finite_library_convergence': stop and not numerical_exclusions and scope == 'finite_library',
        'material_archive_progress': bool(material_progress),
        'hypothesis_material_change': bool(hypothesis_changed),
        'post_selection_review_pending': bool(post_selection_review_pending),
        'evaluation_complete_under_solver_policy': complete,
        'pending_candidates': pending_candidates,
        'numerical_exclusions': numerical_exclusions,
        'global_symbolic_optimality_claimed': False,
        'note': 'A solver exclusion remains an unresolved numerical model, not a mathematical disproof. Unhelpful forms do not reset stability unless they change the selected equation, whose response still requires next-cycle scientific review.'
    }
