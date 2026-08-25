# Frozen protocol: costly deduction points with hidden defectors

Frozen before the live smoke and substantive runs on 2026-08-21.

## Question

Do ordinary GPT-5 Nano agents use a costly sanction selectively against hidden
mechanical defectors, and does access to that sanction change ordinary giving,
returning, or myth content?

## Design

Run a matched 0%-versus-25%-defector Myth→Game screen in eight-agent rotating
populations. Both arms use GPT-5 Nano, ten rounds, two hidden mechanical
defectors in the treatment, signed informed `U(−1,+1)` transfer noise, balanced
anonymous pairings, and normal myth authorship by all agents. Mechanical
defectors always send, return, and deduct zero without game-decision LLM calls.

After the receiver returns, add a neutral deduction-point stage based on the
sender-only punishment design used in trust-game experiments:

- the sender receives two deduction points;
- it may spend 0, 1, or 2 whole points;
- unspent points are added to the sender's round earnings;
- each point spent removes up to three payoff units from the receiver, capped
  so the receiver's round payoff cannot become negative; and
- the receiver is explicitly notified of the choice and payoff loss.

The rules use the neutral term *deduction points* rather than moralizing labels.
The sender observes the noisy/communicated return, not the hidden true return.
The stage is known before play, so it can deter as well as respond to defection.

This follows three established design choices: sender action after observing the
return and a 1:3 cost-to-effect ratio in a trust-game punishment stage
([Calabuig et al., 2016](https://doi.org/10.1016/j.joep.2016.09.006)); a
neutral *deduction points* frame and the same 1:3 ratio in second- and
third-party punishment experiments
([Fehr & Fischbacher, 2004](https://doi.org/10.1016/S1090-5138(04)00005-4));
and explicit agnosticism about whether sanctions help, since their perceived
fairness can preserve or crowd out cooperation
([Fehr & Rockenbach, 2003](https://doi.org/10.1038/nature01474)).

Each agent receives exactly one post-game memory exchange per round: a sender
decision or a scripted receiver outcome notice. Together with one myth and one
game decision, `memory_capacity: 9` preserves the same three-complete-round
horizon as the existing Myth→Game design.

Replicate 60 is a live smoke and is excluded from inference. If both smoke
populations pass, run five new matched population pairs using replicate IDs
61–65.

## Frozen screen outcomes

Primary descriptive mechanism outcomes:

1. deduction points spent by ordinary senders when the receiver is a defector
   versus when the receiver is ordinary;
2. the probability of any deduction in those two target classes; and
3. the relationship between visible return ratio and deduction spending.

Matched treatment diagnostics:

- ordinary-agent sending and returning in 25% versus 0% defector populations;
- subsequent change in a punished receiver's behavior when next acting as a
  sender or receiver;
- targeted versus antisocial deductions (including deductions after a return
  consistent with an equal split); and
- punishment, threat, cooperation, and fairness language in ordinary- and
  defector-authored myths.

This is a mechanism screen, not a powered confirmatory test. Population means
and paired intervals will be reported without promoting a post-hoc endpoint to
confirmatory status.

## Acceptance gate

- complete rounds, pairings, game decisions, myths, deduction decisions, and
  receiver notices;
- exactly one post-game exchange per agent-round;
- every scripted defector game/deduction action equals zero, while defectors
  retain normal myth-writing calls;
- every ordinary sender deduction is an integer in `[0,2]`;
- sender bonus, target loss, payoff floor, cumulative balances, and 1:3 effect
  reconstructed exactly from saved records;
- signed send/return noise within bounds and applied only after each underlying
  decision;
- matched schedules, noise seeds, and defector assignments where applicable;
- no hidden defector label or true noisy transfer leaked to ordinary prompts;
- three complete prior rounds retained under the nine-exchange memory window;
  and
- no unrecovered response-boundary or provider errors.

## Interpretation boundary

Because the deduction opportunity is known from round one, this design jointly
captures deterrence and realized punishment. It does not isolate third-party
punishment: only the directly affected sender may deduct from the receiver.
The two-point budget adds a fixed opportunity endowment to sender payoffs, so
raw final balances are not directly comparable to earlier no-deduction runs.
