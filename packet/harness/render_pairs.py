"""Render consecutive Anthropic requests of one session to token ids with vLLM's own converter + the model's chat
template, and report where request B stops being an extension of request A. Prints counts only, never text."""
import json, sys
from transformers import AutoTokenizer
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages as S

tok = AutoTokenizer.from_pretrained("/model")
merge = S._detect_merge_inline_system(tok.chat_template)


def render(path):
    req = AnthropicMessagesRequest(**json.load(open(path)))
    cc = S.to_chat_completion_request(req, merge_inline_system=merge)
    msgs = [m if isinstance(m, dict) else m.model_dump(exclude_none=True) for m in cc.messages]
    for m in msgs:  # vLLM hands the template parsed arguments (chat_utils), not the JSON string
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"].get("arguments"), str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"] or "{}")
    tools = [t if isinstance(t, dict) else t.model_dump(exclude_none=True) for t in (cc.tools or [])]
    kw = dict(cc.chat_template_kwargs or {})
    ids = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=True, **kw)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    return list(ids), msgs


for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    A, ma = render(a)
    B, mb = render(b)
    i = 0
    while i < min(len(A), len(B)) and A[i] == B[i]:
        i += 1
    roles = [m["role"] for m in mb[len(ma) - 3:]]
    ra = sum(1 for m in ma if m.get("reasoning")); rb = sum(1 for m in mb if m.get("reasoning"))
    print(f"A {len(A)} tok, B {len(B)} tok, common prefix {i} ({i / len(A) * 100:.1f}% of A); "
          f"msgs with reasoning A {ra} B {rb}; B tail roles {roles}")
