"""Shared, frozen coordinate and symbol contract for generation and computation."""
import json
from pathlib import Path
import numpy as np
import sympy as sp

REGISTRY = json.loads((Path(__file__).resolve().parents[1]/'inputs/symbol_registry.json').read_text(encoding='utf-8'))
VARIABLES = tuple(REGISTRY['variables'])
REFERENCE = {name: row['reference'] for name, row in REGISTRY['variables'].items()}


def variable_symbols():
    return {name: sp.Symbol(name, **{row['assumption']: True})
            for name, row in REGISTRY['variables'].items()}


def coordinates(table):
    return {name: (np.asarray(table[row['raw']], dtype=float)+row['offset'])/row['scale']
            for name, row in REGISTRY['variables'].items()}


def validate_inputs(inputs, protocol):
    saved = json.loads((Path(inputs)/'symbol_registry.json').read_text(encoding='utf-8-sig'))
    if saved != REGISTRY:
        raise ValueError('SYMBOL_REGISTRY_DIFFERS_FROM_ALGORITHM')
    if protocol['domains'] != {name: row['domain'] for name, row in REGISTRY['variables'].items()}:
        raise ValueError('SYMBOL_REGISTRY_DOMAIN_MISMATCH')
    if protocol['reference'] != REFERENCE:
        raise ValueError('SYMBOL_REGISTRY_REFERENCE_MISMATCH')
    if set(protocol['grammar']['unary']) != set(REGISTRY['functions']):
        raise ValueError('SYMBOL_REGISTRY_FUNCTION_MISMATCH')
