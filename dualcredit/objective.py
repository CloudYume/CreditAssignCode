import torch


def gather_logprob(logits: torch.Tensor, rows: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    if rows.numel() == 0:
        return torch.zeros(0)
    return log_probs[rows, tokens]


def answer_channel_term(new_logp: torch.Tensor, old_logp: torch.Tensor, advantage: torch.Tensor, delta: float) -> torch.Tensor:
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - delta, 1.0 + delta)
    return -torch.minimum(ratio * advantage, clipped * advantage)


def response_channel_term(new_logp: torch.Tensor, old_logp: torch.Tensor, advantage: torch.Tensor, delta: float | None = None) -> torch.Tensor:
    ratio = torch.exp(new_logp - old_logp)
    if delta is None:
        return -ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - delta, 1.0 + delta)
    return -torch.minimum(ratio * advantage, clipped * advantage)


def kl_k3(new_logp: torch.Tensor, reference_logp: torch.Tensor) -> torch.Tensor:
    diff = reference_logp - new_logp
    return torch.exp(diff) - diff - 1.0


def entropy_term(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(torch.exp(log_probs) * log_probs).sum(dim=-1)


def channel_rows(positions: list, response_mask: torch.Tensor) -> tuple:
    index = torch.arange(len(positions))
    call_flags = torch.tensor([int(position.slot == "call") for position in positions], dtype=torch.bool)
    decision_flags = torch.tensor([int(position.slot == "decision") for position in positions], dtype=torch.bool)
    answer_flags = torch.tensor([int(position.slot == "answer") for position in positions], dtype=torch.bool)
    answer_rows = index[~response_mask]
    response_rows = index[response_mask]
    call_rows = index[call_flags]
    decision_rows = index[decision_flags]
    answer_slots = index[answer_flags]
    return answer_rows, response_rows, call_rows, decision_rows, answer_slots


def dual_channel_loss(new_call: torch.Tensor, new_decision: torch.Tensor, new_answer: torch.Tensor,
                      ref_call: torch.Tensor, ref_decision: torch.Tensor, ref_answer: torch.Tensor,
                      positions: list, advantages: torch.Tensor, response_mask: torch.Tensor,
                      delta: float, kl_beta: float, response_delta: float | None = None, entropy_beta: float = 0.0) -> dict:
    answer_rows, response_rows, call_rows, decision_rows, answer_slots = channel_rows(positions, response_mask)
    new_logp = torch.zeros(len(positions))
    reference_logp = torch.zeros(len(positions))
    tokens = torch.tensor([position.token for position in positions], dtype=torch.long)
    old_logp = torch.tensor([position.logprob for position in positions], dtype=torch.float32)
    if call_rows.numel() > 0:
        new_logp[call_rows] = gather_logprob(new_call, call_rows, tokens[call_rows])
        reference_logp[call_rows] = gather_logprob(ref_call, call_rows, tokens[call_rows])
    if decision_rows.numel() > 0:
        new_logp[decision_rows] = gather_logprob(new_decision, decision_rows, tokens[decision_rows])
        reference_logp[decision_rows] = gather_logprob(ref_decision, decision_rows, tokens[decision_rows])
    if answer_slots.numel() > 0:
        new_logp[answer_slots] = gather_logprob(new_answer, answer_slots, tokens[answer_slots])
        reference_logp[answer_slots] = gather_logprob(ref_answer, answer_slots, tokens[answer_slots])
    answer_loss = answer_channel_term(new_logp[answer_rows], old_logp[answer_rows], advantages[answer_rows], delta) if answer_rows.numel() > 0 else torch.zeros(0)
    response_loss = response_channel_term(new_logp[response_rows], old_logp[response_rows], advantages[response_rows], response_delta) if response_rows.numel() > 0 else torch.zeros(0)
    divergence = kl_k3(new_logp[answer_slots], reference_logp[answer_slots]) if answer_slots.numel() > 0 else torch.zeros(0)
    decision_divergence = kl_k3(new_logp[decision_rows], reference_logp[decision_rows]) if decision_rows.numel() > 0 else torch.zeros(0)
    blocks = [block for block in (divergence, decision_divergence) if block.numel() > 0]
    divergence_all = torch.cat(blocks) if blocks else torch.zeros(0)
    entropy = entropy_term(new_decision[decision_rows]) if decision_rows.numel() > 0 else torch.zeros(0)
    total = answer_loss.mean() if answer_loss.numel() > 0 else torch.tensor(0.0)
    if response_loss.numel() > 0:
        total = total + response_loss.mean()
    if divergence_all.numel() > 0:
        total = total + kl_beta * divergence_all.mean()
    if entropy_beta > 0.0 and entropy.numel() > 0:
        total = total - entropy_beta * entropy.mean()
    return {
        "loss": total,
        "answer_loss": answer_loss.mean() if answer_loss.numel() > 0 else torch.tensor(0.0),
        "response_loss": response_loss.mean() if response_loss.numel() > 0 else torch.tensor(0.0),
        "kl": divergence_all.mean() if divergence_all.numel() > 0 else torch.tensor(0.0),
        "entropy": entropy.mean() if entropy.numel() > 0 else torch.tensor(0.0),
        "new_logp": new_logp,
        "old_logp": old_logp,
        "answer_rows": answer_rows,
        "response_rows": response_rows,
        "shift": (new_logp[response_rows] - old_logp[response_rows]).mean() if response_rows.numel() > 0 else torch.tensor(0.0),
    }
