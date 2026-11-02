"""Versioned dimensionless macros lowered into the shared mathematical grammar.

No alternate evaluator: derivatives, fitting and export see the same expansion.
"""
import ast
import math

VERSION = 'DOMAIN_OPERATORS_1'
EPSILON = 1e-8
CATALOG = {
    'safe_log_pos': {'arity': 1, 'domain': 'argument >= 0',
                     'expansion': 'log(z + epsilon)', 'regularized': True},
    'safe_div_pos': {'arity': 2, 'domain': 'denominator >= 0',
                     'expansion': 'a / (b + epsilon)', 'regularized': True},
    'positive_offset': {'arity': 1, 'domain': 'argument >= 0',
                        'expansion': 'z + epsilon', 'regularized': True},
}

def lower(tree, epsilon=EPSILON):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('EPSILON_MUST_BE_FINITE_POSITIVE')
    obligations, receipts = [], []
    class Lower(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if not isinstance(node.func, ast.Name) or node.func.id not in CATALOG:
                return node
            name = node.func.id
            if node.keywords or len(node.args) != CATALOG[name]['arity']:
                raise ValueError('OPERATOR_ARITY: '+name)
            z = node.args[-1]
            obligations.append({'expression': ast.unparse(z), 'relation': 'GE_ZERO',
                                'reason': name+' requires a nonnegative input; epsilon cannot repair negative inputs.'})
            shifted = ast.BinOp(left=z, op=ast.Add(), right=ast.Constant(epsilon))
            if name == 'safe_log_pos':
                result = ast.Call(func=ast.Name(id='log', ctx=ast.Load()), args=[shifted], keywords=[])
            elif name == 'safe_div_pos':
                result = ast.BinOp(left=node.args[0], op=ast.Div(), right=shifted)
            else:
                result = shifted
            receipts.append({'operator': name, 'epsilon': epsilon,
                             'expanded': ast.unparse(ast.fix_missing_locations(result))})
            return ast.copy_location(result, node)
    return ast.fix_missing_locations(Lower().visit(tree)), obligations, receipts
