# TrumpMaster — Canonical Rules

**Ruleset Version:** 2.0.0
**Status:** Current / authoritative
**Matches engine:** `TrumpMaster_v0.3.0_Build03`
**Verified by:** `verify-rules.js` (23/23 scenarios) · match reproducibility verified by `replay-verify.js`

> This is the single source of truth for *what the game is*. Every rule below is
> asserted by a test in `verify-rules.js`. If code and this document ever
> disagree, that is a bug in one of them — not a matter of opinion. Keep them in
> lockstep.

---

## 1. Components

- One standard **52-card** deck. No jokers.
- Ranks low→high: `2 3 4 5 6 7 8 9 10 J Q K A`. **Ace is always highest.**
- Exactly **four** players. Seats are fixed:

| Seat | Position | Default |
|------|----------|---------|
| 0 | South | Human |
| 1 | East | Bot |
| 2 | North | Bot |
| 3 | West | Bot |

The engine operates on **seat numbers only**; compass names are presentation.

## 2. Turn order

`Seat 0 → Seat 1 → Seat 2 → Seat 3 → Seat 0`

This single order governs dealing, bidding, card play, and dealer rotation.

## 3. Dealing

- Each player receives exactly **13 cards**.
- The **first** card goes to the seat immediately **after the dealer**.
- Dealing proceeds in turn order; the **dealer receives the 52nd (final) card**.
- After each deal the dealer rotates **+1 seat**. Over the 8 deals of a game,
  each seat is dealer **exactly twice**.

## 4. Trump

- The dealer's final (52nd) card sets the **trump suit**.
- It is **revealed after dealing** and stays visible for the whole deal.
- Trump is **fixed** until the deal ends.

## 5. Analysis phase

- Begins immediately after the trump reveal and lasts **20 seconds**.
- Players may inspect **only their own hand and the revealed trump**.
- **No** cards may be played and **no** bids may be made during it.

## 6. Bidding

- Begins after the analysis phase.
- Order: the seat **after the dealer bids first**; the **dealer bids last**.
- Each player bids **exactly once**. Valid bids are **2 to 13 inclusive**.
- All bids placed so far remain **visible** throughout bidding.
- *Timeout:* if a player does not bid in time, the **minimum bid (2)** is recorded.

## 7. Play

- The leader of the **first trick** is the seat **after the dealer**.
- For every later trick, the **winner of the previous trick leads**.
- Play proceeds in turn order until all four players have played one card.
- *Timeout:* if a player does not act in time, an **automatic legal card** is
  played on their behalf.
- Legality is decided **only** by the hierarchy in Article 8.

## 8. Legal-play hierarchy  *(normative — evaluate top to bottom; first match wins)*

**1. Leading the trick** (no card played yet)
→ any card may be played.

**2. The lead suit IS trump, and the player holds at least one trump**
- If the player can play a trump **higher** than the highest trump in the trick,
  they **must** (overtrump).
- Otherwise they may play **any** trump they hold.

**3. The lead suit is NOT trump, and the player holds the lead suit**
- **(a) A trump has already been played to this trick:**
  the player must follow the lead suit, but may play **any** card of it.
  *(No must-beat obligation — no lead-suit card can win a trick that is already
  trumped.)*
- **(b) No trump has been played to this trick:**
  - If the player can beat the **highest lead-suit card** in the trick, they
    **must** play a higher card of the lead suit.
  - Otherwise they must still play **some** card of the lead suit.

**4. The player does NOT hold the lead suit**
- Holds **no trump** → any card.
- Holds trump, **none played yet** → **must play a trump**.
- Holds trump, **trump already played:**
  - Can overtrump → **must** play a higher trump.
  - Cannot overtrump → any card, **including a lower trump**.

> **Design note.** Rules 2 and 3(b) make TrumpMaster a *must-head-the-trick*
> game whenever heading it is still possible. This is deliberate and stricter
> than some Callbreak variants. Rule 3(a) is the one carve-out: once a trick is
> trumped, heading it is impossible, so the obligation lifts.

## 9. Trick resolution

1. If any trump was played, the **highest trump** wins.
2. Otherwise, the **highest card of the lead suit** wins.

The winner immediately becomes the leader of the next trick.

## 10. Deal & game completion

- A deal ends after **13 tricks**. Scores are then calculated (Article 11).
- The dealer rotates and a new deal begins.
- The **game ends after 8 deals**; final ranking is applied (Article 12).

## 11. Scoring  *(per player, per deal — bid `n`, tricks won `m`)*

| Condition | Score |
|-----------|-------|
| `m = n` (exact) | `+n` |
| `m < n` (under) | `−n` |
| `n < m < 2n` | `n + (m − n) / 10` |
| `m ≥ 2n` | `−n` |

The boundary `m = 2n` scores `−n` (it belongs to the `≥ 2n` row, not the
between row).

## 12. Tie-breaking / final ranking

Rank players by, in order:
1. Highest **total score**.
2. **Fewest** negative deals.
3. **Lowest** cumulative thinking time.

---

## Appendix A — The changed rule, worked

This is the exact case that separates Ruleset **1.0** from Ruleset **2.0**.

```
Trump: ♠

South leads   ♥K        (hearts, a non-trump suit)
East plays    ♠5        (East was void in hearts, so it ruffed)
North to act — North still holds ♥A and ♥3
```

North holds the lead suit, so North must play a heart. The question is *which*.

| | North's legal cards | Why |
|---|---|---|
| **Ruleset 1.0** (old prototype) | `♥A` only | Must beat the ♥K, even though the trick is already lost to ♠5 |
| **Ruleset 2.0** (current) | `♥A` or `♥3` | Article 8, rule **3(a)**: the trick is already trumped, so the must-beat obligation lifts |

Ruleset 2.0 is current and correct: forcing North to burn the Ace of Hearts on a
trick no heart can win is punishing and non-standard. Both branches are pinned in
`verify-rules.js` as the *BEFORE ruff → A only* and *AFTER ruff → A or 3* tests.

---

## Appendix B — Determinism & replay (Build 03)

**Resolved as of Build 03.** The engine now guarantees: given the same match
seed and the same sequence of bids/plays, the outcome is always identical.

- **Shuffle.** `Deck` takes a seeded PRNG (`mulberry32`) instead of
  `Math.random()`. The seed is drawn once per match from a true entropy
  source (`crypto.getRandomValues`, with a `Math.random()` fallback) and
  displayed to the player as an 8-character hex label (e.g. `DEADBEEF`).
- **Replay log.** Every bid and card play is recorded as a minimal event
  (`bid:placed`, `card:played`, `deal:scored`, plus `deal:start` markers),
  each stamped with the actual thinking-time elapsed. This — not a re-drawn
  random number — is what makes bot "thinking delay" reproducible in tie
  scoring: the log stores what happened, not how to re-derive it.
- **Export.** At game end, `Download Replay JSON` produces a file containing
  `matchSeed`, `rulesetVersion` (`2.0.0`), `replayFormatVersion` (`1.0.0`),
  the full event log, and the final scores/ranking.
- **Independent verification.** `replay-verify.js` takes a replay file,
  regenerates every deal from the seed alone, re-validates that every
  recorded play was legal at the moment it was made
  (`RuleEngine.getLegalCards`), recomputes every trick winner and deal
  score from scratch, and diffs the result against what the replay recorded.
  Any mismatch — an illegal move, a tampered card, a scoring error — is
  reported explicitly rather than silently trusted.
- **Headless simulation.** `simulate-match.js` runs a full 8-deal match with
  no browser, using the same `RuleEngine`/`Deck` logic, and writes a replay
  file — useful for regression-testing the engine itself and for generating
  test fixtures.

**What this gives you in practice:** a bug report is now "here's my
`trumpmaster_replay_XXXXXXXX.json`" — `node replay-verify.js <file>`
reproduces the exact match and tells you precisely where, if anywhere, the
recorded game deviated from a legal one.

**Still open:** the replay format does not yet capture client-side timing
jitter or network conditions (irrelevant for a local prototype, but will
matter once this moves server-authoritative — see the platform architecture
discussion). Multiplayer fairness (proving the *server's* RNG wasn't
manipulated) is a separate, larger problem this replay format does not
attempt to solve on its own.

---

## Changelog

**Rules** (this is what `rulesetVersion` tracks — gameplay laws only):

| Ruleset | Change |
|---------|--------|
| 2.0.0 | Article 8 rule **3(a)** added: after a trick is trumped, a lead-suit holder may follow with any card of the suit (no forced beat). |
| 1.0.0 | Original: a lead-suit holder must always beat the highest lead-suit card if able, regardless of whether the trick had been trumped. |

**Engine** (does not change gameplay rules, so `rulesetVersion` stays `2.0.0`):

| Build | Change |
|-------|--------|
| Build 03 | Seeded RNG (`mulberry32`), replay recording + JSON export, and an independent `replay-verify.js` verifier. Closes the determinism gap from Appendix B. |
| Build 02 | Rule 3(a) implemented; engine and rulebook reconciled. |
