"""Bounded clarification of contradictory scientific reviews, never auto-waivers."""
from . import generation
from .storage import save

INSTRUCTIONS = """Review every supplied proposal ID exactly once.
Separate demonstrated mathematical/source/domain errors from qualified modelling
assumptions. An honestly labelled macroscopic closure need not be a published
exact formula. Lack of an identical source law, limited extrapolation, or an
untested assumption alone belongs in caveats, not critical_issues. A false
identity, unsupported source attribution, impossible domain, or an incoherent
process-to-math argument is critical and must not be waived.
Keep the verdict internally consistent: ELIGIBLE or ELIGIBLE_WITH_CAVEATS
requires an empty critical_issues list. A blocking issue requires QUARANTINE
and a concrete critical_issues explanation. Mathematical probes are not proof
of a physical mechanism. No desired equation or parameter count is supplied.
Check question_links and contrast_links against the supplied question/contrast
contracts, not just their IDs: does the proposed law/rival express the stated
observable distinction under its conditions? Wrong scope, response or a falsely
claimed correspondence is blocking. A valid proposed correspondence is not
confirmation of the hypothesis. Missing measurements remain caveats/next tests.
Use symbol_compilation and compiled_proposal in the mathematical checks to distinguish
declared intermediate aliases from truly undeclared symbols. Algebraic expansion does
not prove that the source mapping, units, constants or prefactor normalization are
scientifically justified; review these explicitly. Reference-condition failure is
not nonfinite algebra. conditions are global feasibility constraints, whereas local
derivative signs and peak-branch statements belong in trend_scope.
"""


def coverage(review, proposals):
    verdicts = {v['proposal_id']: v for v in review['verdicts']}
    if len(review['verdicts']) != len(proposals) or set(verdicts) != {p['proposal_id'] for p in proposals}:
        raise RuntimeError('CRITIC_ID_COVERAGE_MISMATCH')
    return verdicts


def contradictory(verdict):
    return ((verdict['verdict'] != 'QUARANTINE' and bool(verdict['critical_issues'])) or
            (verdict['verdict'] == 'QUARANTINE' and not verdict['critical_issues']))


def reviewed(call, proposals, checks, evidence, folder, initial_review=None, scientific_contracts=None):
    """One clarification at most; both raw judgments remain in immutable call logs.

    Check failures are never cleared by an LLM. A still contradictory review
    remains blocked. A coherent QUARANTINE is never retried to obtain approval.
    """
    _, _, schema, _ = generation.schemas()
    payload = {'instructions': INSTRUCTIONS, 'proposals': proposals,
               'mathematical_checks': checks, 'evidence': evidence,
               'scientific_contracts':scientific_contracts or {}}
    initial = initial_review if initial_review is not None else call(
        'INDEPENDENT_SCIENTIFIC_CRITIC', payload, schema, folder/'initial')
    verdicts = coverage(initial, proposals)
    check_by_id = {p['proposal_id']: check for p, check in zip(proposals, checks)}
    conflicts = [p for p in proposals if not check_by_id[p['proposal_id']]['errors']
                 and contradictory(verdicts[p['proposal_id']])]
    audits = {p['proposal_id']: {'initial_review': verdicts[p['proposal_id']],
              'clarification_requested': False, 'unresolved_conflict': contradictory(verdicts[p['proposal_id']])}
              for p in proposals}
    if conflicts:
        clarified = call('SCIENTIFIC_REVIEW_CONFLICT_CLARIFIER', {
            'instructions': INSTRUCTIONS + '\nClarify only the internal conflicts below. '
                'Reassess the substance: retain genuine blocking issues as QUARANTINE; '
                'move only nonblocking limitations to caveats. Do not merely empty '
                'the critical list to admit a candidate. A genuine mathematical '
                'error cannot be overridden. This is the only clarification pass.',
            'proposals': conflicts,
            'mathematical_checks': [check_by_id[p['proposal_id']] for p in conflicts],
            'original_verdicts': [verdicts[p['proposal_id']] for p in conflicts],
            'evidence': evidence,'scientific_contracts':scientific_contracts or {}}, schema, folder/'clarification')
        replacements = coverage(clarified, conflicts)
        for pid, replacement in replacements.items():
            audits[pid].update(clarification_requested=True, clarified_review=replacement,
                              unresolved_conflict=contradictory(replacement))
            verdicts[pid] = replacement
    save(folder/'resolution.json', {'policy': 'One substantive clarification, no automatic waiver.',
        'mathematical_errors_waivable': False, 'records': audits,
        'clarified_count': len(conflicts),
        'unresolved_count': sum(v['unresolved_conflict'] for v in audits.values())})
    return verdicts, audits
