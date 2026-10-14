"""Benchmark crukx-sync on real GRPO updates (Qwen2.5-1.5B-Instruct, bf16 broadcast per step).

Usage: python bench/grpo.py LR STEPS OUT   (needs torch + transformers + a CUDA GPU)
Each step: encode_update(prev, cur) on the trainer, apply_update on a copy (the receiver),
check the copy equals cur bit-for-bit, and record bytes and times.

Training setup, from crukx-research R1g:

Policy (bf16) on the GPU samples answers to arithmetic prompts; reward = exact answer.
GRPO loss: group-normalised advantage x mean token log-prob of each completion
(no KL term, no clipping: one update per batch). Gradients go to fp32 master weights
with AdamW on the CPU (the 3090 cannot hold fp32 master + Adam for 1.5B); the bf16 cast
of the master weights is written back to the GPU policy: that cast is the broadcast.

Per step, vs the previous broadcast: changed fraction, the old-exponent-context entropy
bound (R1e) and the concrete Rice+Huffman format size (R1f).
"""
import json, os, random, re, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from crukx_sync import apply_update, encode_update
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.expanduser("~/crukx-experiments/20261009-xet/models/Qwen2.5-1.5B-Instruct")
LR = float(sys.argv[1]); STEPS = int(sys.argv[2]); OUT = sys.argv[3]
PROMPTS, GROUP, MAX_NEW = 8, 8, 12
random.seed(0); torch.manual_seed(0); os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
policy = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
policy.gradient_checkpointing_enable()
names = [n for n, _ in policy.named_parameters()]
master = {n: p.detach().float().cpu().clone().requires_grad_(False) for n, p in policy.named_parameters()}
opt = torch.optim.AdamW(list(master.values()), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
total = sum(p.numel() for p in policy.parameters())


def task():
    # mixed difficulty so most groups contain both right and wrong answers
    a, b = random.randint(10, 999), random.randint(10, 999)
    q = f"What is {a} + {b}? Reply with only the number."
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return text, a + b


def reward(out, ans):
    nums = re.findall(r"-?\d+", out.replace(",", ""))
    return 1.0 if nums and nums[-1] == str(ans) else 0.0


def bf16_state():
    """The broadcast: bf16 bit patterns as numpy uint16, per tensor."""
    return {n: p.detach().view(torch.int16).cpu().numpy().view(np.uint16).copy() for n, p in policy.named_parameters()}


prev = bf16_state(); rows = []
for step in range(1, STEPS + 1):
    t0 = time.time()
    tasks = [task() for _ in range(PROMPTS)]
    texts = [t for t, _ in tasks for _ in range(GROUP)]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    policy.eval()
    with torch.no_grad():
        gen = policy.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0,
                              pad_token_id=tok.pad_token_id)
    comp = gen[:, enc.input_ids.shape[1]:]
    outs = tok.batch_decode(comp, skip_special_tokens=True)
    r = torch.tensor([reward(o, tasks[i // GROUP][1]) for i, o in enumerate(outs)])
    rg = r.view(PROMPTS, GROUP)
    adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).flatten().cuda()
    # GRPO loss on completion tokens
    policy.train()
    attn = (gen != tok.pad_token_id).long()
    attn[:, :enc.input_ids.shape[1]] = enc.attention_mask
    logits = policy(input_ids=gen, attention_mask=attn).logits[:, enc.input_ids.shape[1] - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
    cmask = (comp != tok.pad_token_id).float()
    seq_lp = (lp * cmask).sum(1) / cmask.sum(1).clamp(min=1)
    loss = -(adv * seq_lp).mean()
    loss.backward()
    for n, p in policy.named_parameters():
        master[n].grad = p.grad.detach().float().cpu(); p.grad = None
    opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in policy.named_parameters():
            p.copy_(master[n].to(torch.bfloat16))
    cur = bf16_state()
    row = {"step": step, "reward": round(r.mean().item(), 3), "loss": round(loss.item(), 4),
           "step_s": round(time.time() - t0, 1)}
    t1 = time.time(); upd = encode_update(prev, cur); enc_s = time.time() - t1
    recv = {k: v.copy() for k, v in prev.items()}
    t1 = time.time(); stats = apply_update(recv, upd); dec_s = time.time() - t1
    exact = all(np.array_equal(recv[k], cur[k]) for k in cur)
    full = 2 * total
    row.update({"changed_frac": round(stats.changed / total, 5), "update_bytes": len(upd),
                "B_per_change": round(len(upd) / max(stats.changed, 1), 3),
                "full_over_update": round(full / len(upd), 1),
                "verl_over_update": round(6 * stats.changed / len(upd), 2),
                "encode_s": round(enc_s, 2), "apply_s": round(dec_s, 2), "exact": exact})
    assert exact, "receiver does not match the trainer"
    rows.append(row); print(json.dumps(row), flush=True)
    prev = cur
json.dump({"lr": LR, "params": total, "rows": rows}, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
