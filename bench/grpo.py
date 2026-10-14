"""Benchmark crukx-sync on real GRPO updates (bf16 broadcast per step).

Usage: [CRUKX_BENCH_MODEL=...] python bench/grpo.py LR STEPS OUT   (torch + transformers + a CUDA GPU)

Policy (bf16) on the GPU samples answers to 2-3 digit additions; reward = exact answer.
GRPO loss: group-normalised advantage x mean token log-prob of each completion (no KL, no
clipping, one update per batch). Gradients go to fp32 master weights with AdamW on the CPU (a
24 GB GPU cannot hold fp32 master + Adam for 1.5B); the bf16 cast written back to the GPU policy
is the broadcast. Each step: encode_update(prev, cur) on the trainer, apply_update on the
receiver's own tensors (kept across steps, with checksum caches), then a bit-for-bit check.
"""
import json, os, random, re, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from crukx_sync import apply_update, encode_update
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("CRUKX_BENCH_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # HF id or local path
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
    """The broadcast: a GPU copy of the bf16 weights."""
    return {n: p.detach().clone() for n, p in policy.named_parameters()}


prev = bf16_state(); rows = []
send_chk, recv_chk = {}, {}  # checksum caches kept across steps (first step computes them)
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
    torch.cuda.synchronize(); t1 = time.time(); upd = encode_update(prev, cur, send_chk); torch.cuda.synchronize(); enc_s = time.time() - t1
    recv = prev  # the receiver's own tensors, updated in place every step
    t1 = time.time(); stats = apply_update(recv, upd, recv_chk); torch.cuda.synchronize(); dec_s = time.time() - t1
    exact = all(torch.equal(recv[k].view(torch.int16), cur[k].view(torch.int16)) for k in cur)
    full = 2 * total
    row.update({"changed_frac": round(stats.changed / total, 5), "update_bytes": len(upd),
                "B_per_change": round(len(upd) / max(stats.changed, 1), 3),
                "full_over_update": round(full / len(upd), 1),
                "verl_over_update": round(6 * stats.changed / len(upd), 2),
                "encode_s": round(enc_s, 3), "apply_s": round(dec_s, 3), "exact": exact})
    assert exact, "receiver does not match the trainer"
    rows.append(row); print(json.dumps(row), flush=True)
    # `prev` (the receiver's tensors) now equals `cur`; keep the same tensors, as a real worker does
    del cur
json.dump({"lr": LR, "params": total, "rows": rows}, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
