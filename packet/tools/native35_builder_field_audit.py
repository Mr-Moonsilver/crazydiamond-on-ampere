#!/usr/bin/env python3
"""Exact field-by-field audit of the pinned attention metadata builders (CPU).

For every builder class this model's ``runner.attn_groups`` can hold, walk the
pinned ``__init__`` chain with ``ast`` and list each ``self.<field> = ...``
assignment with its source line and the value expression, then classify the
expression the way ``lane_chunk._builder_field_class`` / ``_clone_builder_value``
would classify the constructed value:

* ``private-tensor``      torch.empty/zeros/tensor/arange constructions
* ``private-container``   list/tuple/dict/set literals or ``list(...)`` calls
* ``shared-read-only``    literals, bools/ints/strs, ``None``, enum/dtype reads,
                          and the four names in ``_SHARED_CONFIG_FIELDS``
* ``config-object``       ``vllm_config.<x>_config`` reads outside the shared
                          names: refused by the 2d66c89 classifier
* ``numpy-alias``         ``<tensor>.numpy()``: refused by the classifier and
                          an intra-builder alias of the tensor it views
* ``conditional``         the assignment sits under an ``if``; the branch
                          condition is recorded
* ``unresolved``          anything else (reported, never guessed)

It also sizes the tensor fields from the scheduler/compilation settings
passed on the command line so the per-rank byte total can be bounded before
the box ledger reports the real number. Static only: no torch, no CUDA.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

BUILDERS = {
    # class -> (relative source file, documented base chain in MRO order)
    'QSAMetadataBuilder': ('models/qwen4_exp/common/qsa_cache.py', ('AttentionMetadataBuilder',)),
    'Qwen4ExpQSAMetadataBuilder': ('models/qwen4_exp/nvidia/qsa.py',
                                   ('FlashAttentionMetadataBuilder', 'AttentionMetadataBuilder')),
    'GDNAttentionMetadataBuilder': ('v1/attention/backends/gdn_attn.py', ('AttentionMetadataBuilder',)),
    'PleShortConvAttentionMetadataBuilder': ('v1/attention/backends/short_conv_attn.py',
                                             ('ShortConvAttentionMetadataBuilder',
                                              'BaseMambaAttentionMetadataBuilder',
                                              'AttentionMetadataBuilder')),
}
SOURCES = {
    'AttentionMetadataBuilder': 'v1/attention/backend.py',
    'FlashAttentionMetadataBuilder': 'v1/attention/backends/flash_attn.py',
    'ShortConvAttentionMetadataBuilder': 'v1/attention/backends/short_conv_attn.py',
    'BaseMambaAttentionMetadataBuilder': 'v1/attention/backends/mamba_attn.py',
}
SHARED_CONFIG_FIELDS = {'kv_cache_spec', 'vllm_config', 'device', 'backend'}
TENSOR_CTORS = {'empty', 'zeros', 'ones', 'tensor', 'arange', 'full', 'empty_like', 'zeros_like'}


def _src(node, text):
    return ast.get_source_segment(text, node) or type(node).__name__


def _classify(name, value, text):
    if name in SHARED_CONFIG_FIELDS:
        return 'shared-read-only', 'name in _SHARED_CONFIG_FIELDS'
    if isinstance(value, ast.Call):
        fn = value.func
        if isinstance(fn, ast.Attribute):
            if fn.attr == 'numpy':
                return 'numpy-alias', 'view of ' + _src(fn.value, text)
            if fn.attr in TENSOR_CTORS and isinstance(fn.value, ast.Name) and fn.value.id == 'torch':
                return 'private-tensor', 'torch.' + fn.attr
            if fn.attr in ('has_full_cudagraphs',):
                return 'shared-read-only', 'bool from ' + _src(fn, text)
        if isinstance(fn, ast.Name):
            if fn.id in ('list', 'tuple', 'dict', 'set'):
                return 'private-container', fn.id + '(...)'
            if fn.id in ('int', 'bool', 'str', 'min', 'max', 'cdiv', 'round_up', 'len'):
                return 'shared-read-only', fn.id + '(...) scalar'
            if fn.id in ('isinstance', 'getattr'):
                return 'shared-read-only', fn.id + '(...)'
        return 'unresolved', 'call ' + _src(fn, text)
    if isinstance(value, ast.Constant):
        return 'shared-read-only', repr(value.value)
    if isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return 'private-container', 'literal'
    if isinstance(value, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
        return 'shared-read-only', 'bool expression'
    if isinstance(value, ast.BinOp):
        return 'shared-read-only', 'arithmetic'
    if isinstance(value, ast.Attribute):
        chain = _src(value, text)
        if chain.endswith('_config') and ('vllm_config' in chain or chain.startswith('self.vllm_config')):
            return 'config-object', chain
        if chain.startswith('self.') and chain.count('.') == 1:
            return 'shared-read-only', 'copy of own scalar ' + chain
        # kv_cache_spec.dtype / .num_kv_heads / .block_size, scheduler ints,
        # model_config.rswa_window, speculative ints: scalar or dtype reads.
        return 'shared-read-only', 'scalar/dtype read ' + chain
    if isinstance(value, ast.Name):
        return 'shared-read-only', 'local ' + value.id
    if isinstance(value, ast.IfExp):
        return 'shared-read-only', 'conditional scalar'
    return 'unresolved', _src(value, text)


def _walk_init(cls_node, text):
    init = next((n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
    if init is None:
        return None
    rows = []

    def visit(nodes, conditions):
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                            and target.value.id == 'self' and node.value is not None):
                        cls, why = _classify(target.attr, node.value, text)
                        rows.append(dict(field=target.attr, line=node.lineno, cls=cls, why=why,
                                         value=_src(node.value, text).replace('\n', ' ')[:160],
                                         conditions=list(conditions)))
            elif isinstance(node, ast.If):
                cond = _src(node.test, text).replace('\n', ' ')[:120]
                visit(node.body, conditions + ['if ' + cond])
                visit(node.orelse, conditions + ['else ' + cond])
            elif isinstance(node, ast.Try):
                visit(node.body, conditions + ['try'])
                for handler in node.handlers:
                    visit(handler.body, conditions + ['except'])
            elif isinstance(node, (ast.With, ast.For, ast.While)):
                visit(node.body, conditions + [type(node).__name__.lower()])
    visit(init.body, [])
    return rows


def _class(root, rel, name):
    path = Path(root) / rel
    if not path.exists():
        return None, None
    text = path.read_text()
    tree = ast.parse(text)
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None)
    return node, text


def tensor_bytes(field, settings):
    """Element counts from the pinned __init__ shapes under the given settings."""
    T, S, spec, cap = settings['max_num_batched_tokens'], settings['max_num_seqs'], settings['num_spec'], settings['max_capture']
    bs = min(S * (spec + 1), cap) if cap else S * (spec + 1)
    bs_mamba = min(S, cap) if cap else S
    table = {
        # QSAMetadataBuilder
        'token_to_req_buffer': T * 4, 'slot_mapping_buffer': T * 8, 'logical_positions_buffer': T * 8,
        'visible_blocks_buffer': T * 4,
        'k_work_metadata_buffer': ((T + (settings['compress_ratio'] - 1) * S) // settings['compress_ratio']) * 2 * 4,
        # GDNAttentionMetadataBuilder (decode_cudagraph_max_bs = min(S*(spec+1), cap))
        'spec_state_indices_tensor': bs * (spec + 1) * 4, 'non_spec_state_indices_tensor': bs * 4,
        'spec_sequence_masks': bs * 1, 'spec_token_indx': bs * (spec + 1) * 4,
        'non_spec_token_indx': bs * (spec + 1) * 4, 'spec_query_start_loc': (bs + 1) * 4,
        'non_spec_query_start_loc': (bs + 1) * 4, 'num_accepted_tokens': bs * 4,
        # BaseMamba / PLE (mamba_cache_mode != "all": state_indices_tensor_d is (bs, 1+spec))
        'state_indices_tensor_d': bs_mamba * (1 + spec) * 4, 'has_initial_states_d': bs_mamba * 1,
        'decode_num_accepted_tokens': bs_mamba * 4,
    }
    return table.get(field)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm-root', required=True)
    parser.add_argument('--out')
    parser.add_argument('--max-num-batched-tokens', type=int, default=1024)
    parser.add_argument('--max-num-seqs', type=int, default=10)
    parser.add_argument('--num-spec', type=int, default=0)
    parser.add_argument('--max-capture', type=int, default=24)
    parser.add_argument('--compress-ratio', type=int, default=4)
    args = parser.parse_args(argv)
    settings = dict(max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=args.max_num_seqs,
                    num_spec=args.num_spec, max_capture=args.max_capture, compress_ratio=args.compress_ratio)
    report = dict(schema='native35-builder-field-audit/1', vllm_root=str(args.vllm_root), settings=settings,
                  builders={}, missing=[], refused_by_2d66c89={}, aliases={})
    for name, (rel, bases) in BUILDERS.items():
        chain = [(name, rel)] + [(base, SOURCES[base]) for base in bases]
        rows = []
        for cls_name, cls_rel in chain:
            node, text = _class(args.vllm_root, cls_rel, cls_name)
            if node is None:
                report['missing'].append(f'{name}: {cls_name} not found in {cls_rel}')
                continue
            for row in _walk_init(node, text) or []:
                row['declared_in'] = cls_name + ' ' + cls_rel
                nbytes = tensor_bytes(row['field'], settings) if row['cls'] == 'private-tensor' else None
                row['bytes_under_settings'] = nbytes
                rows.append(row)
        # Later assignments to the same field override; keep the full history.
        fields = {}
        for row in rows:
            fields.setdefault(row['field'], []).append(row)
        refused = sorted(f for f, hist in fields.items()
                         if any(r['cls'] in ('config-object', 'numpy-alias', 'unresolved') for r in hist))
        aliases = sorted(f for f, hist in fields.items() if any(r['cls'] == 'numpy-alias' for r in hist))
        unconditional_tensor_bytes = sum(
            (r['bytes_under_settings'] or 0) for hist in fields.values() for r in hist
            if r['cls'] == 'private-tensor' and not r['conditions'])
        report['builders'][name] = dict(chain=[c for c, _ in chain], fields=fields,
                                        refused_fields=refused, numpy_alias_fields=aliases,
                                        unconditional_tensor_bytes=unconditional_tensor_bytes)
        report['refused_by_2d66c89'][name] = refused
        report['aliases'][name] = aliases
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + '\n')
    for name, b in report['builders'].items():
        print(f"{name}: fields {len(b['fields'])} refused {b['refused_fields']} aliases {b['numpy_alias_fields']} "
              f"unconditional tensor bytes {b['unconditional_tensor_bytes']}")
    for row in report['missing']:
        print('MISSING', row)
    return 0 if not report['missing'] else 2


if __name__ == '__main__':
    sys.exit(main())
