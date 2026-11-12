"""Explicit acyclic aliases; no guessed notation, added fit parameters or domain edits."""
import ast
import copy
import re


def compile_proposal(proposal):
    from .construction import safe_tree
    from .symbols import REGISTRY
    p = copy.deepcopy(proposal)
    parameters = {x['name'] for x in p['parameters']}
    reserved = set(REGISTRY['parameter_contract']['reserved_names'])
    if parameters & reserved or len(parameters) != len(p['parameters']):
        raise ValueError('Invalid or duplicate parameter names')
    definitions = p.get('symbol_definitions', [])
    names = [d['name'] for d in definitions]
    if len(names) != len(set(names)) or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', n)
            or n in reserved or n in parameters for n in names):
        raise ValueError('SYMBOL_DEFINITION_COLLISION_OR_INVALID_NAME')
    by_name = {d['name']: d for d in definitions}
    expanded, active, graph = {}, set(), []

    class Expand(ast.NodeTransformer):
        def visit_Name(self, node):
            return copy.deepcopy(resolve(node.id)) if node.id in by_name else node

    def parse(text, field):
        try:
            tree = safe_tree(text, parameters | set(names))
            node = Expand().visit(tree)
            result = ast.unparse(ast.fix_missing_locations(node))
            safe_tree(result, parameters)
            return result
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f'{field}: {exc}') from exc

    def resolve(name):
        if name in expanded:
            return expanded[name]
        if name in active:
            raise ValueError('CYCLIC_SYMBOL_DEFINITION: '+name)
        active.add(name)
        d = by_name[name]
        if not d.get('meaning', '').strip():
            raise ValueError('SYMBOL_DEFINITION_MISSING_MEANING: '+name)
        expr = parse(d['expression'], 'symbol_definitions.'+name)
        expanded[name] = ast.parse(expr, mode='eval').body
        active.remove(name)
        graph.append({'symbol': name, 'definition': d['expression'], 'expanded': expr,
                      'depends_on': sorted({n.id for n in ast.walk(ast.parse(d['expression'], mode='eval'))
                                            if isinstance(n, ast.Name) and n.id in by_name})})
        return expanded[name]

    for name in names:
        resolve(name)
    # The final fitted expression always exposes its real variables/parameters.
    safe_tree(p['expression'], parameters)
    for field in ('rate_expression_before_log', 'claimed_derivative'):
        if p[field].strip():
            p[field] = parse(p[field], field)
    for index, step in enumerate(p['construction_steps']):
        step['expression_after_step'] = parse(step['expression_after_step'], f'Construction step {index}')
    for index, cond in enumerate(p['conditions']):
        cond['expression'] = parse(cond['expression'], f'conditions.{index}')
    return p, {'version': 'V6_SYMBOL_COMPILER_1', 'definitions': graph,
               'source_symbols': p.get('source_symbols', []),
               'scope': 'Explicit algebraic substitution only; source mappings and units require scientific review. No unit equivalence inferred.'}
