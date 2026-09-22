/* ============================================================================
   replay-verify.js
   Given a replay JSON (as produced by simulate-match.js or the HTML build's
   "Download Replay" button), independently reconstructs the entire match
   from scratch — using only matchSeed + the recorded bid/card decisions —
   and checks:

     1. The reconstructed deck/hands/trump for every deal matches what the
        seed deterministically produces (implicit: we regenerate them here).
     2. Every recorded card play was LEGAL at the moment it was played
        (RuleEngine.validateMove), given the actual hand and trick state.
     3. Every recorded trick winner and deal score is independently
        recomputed and diffed against what happened.
     4. The final scores and ranking in the replay header match what we
        recompute.

   This is the thing that actually closes the "determinism" gap from
   RULES.md Appendix B: a bug report is now "here's my replay.json", and
   this script either reproduces the exact reported state or tells you
   exactly where the recorded match deviates from a legal one.

   Usage: node replay-verify.js <replay.json>
============================================================================ */
const fs = require("fs");
const { RuleEngine } = require("./rule-engine.js");
const { Deck, TurnManager } = require("./deck-engine.js");
const { mulberry32, labelToSeed } = require("./seeded-rng.js");

function cardId(c) { return `${c.rank}${c.suit}`; }

function verifyReplay(replay) {
  const problems = [];
  const seed = labelToSeed(replay.matchSeed);
  const rng = mulberry32(seed);

  const players = [0, 1, 2, 3].map(seat => ({
    seat, bid: null, tricksWon: 0, score: 0, negativeDealCount: 0, thinkTimeMs: 0
  }));

  // Index events per deal for easy lookup.
  const dealStarts = replay.events.filter(e => e.type === "deal:start");
  const bidsByDeal = {}, playsByDeal = {}, scoredByDeal = {};
  replay.events.forEach(e => {
    if (e.type === "bid:placed") (bidsByDeal[e.dealNumber] ||= []).push(e);
    if (e.type === "card:played") (playsByDeal[e.dealNumber] ||= []).push(e);
    if (e.type === "deal:scored") scoredByDeal[e.dealNumber] = e;
    if (e.type === "card:played") players[e.seat].thinkTimeMs += (e.elapsedMs || 0);
    if (e.type === "bid:placed") players[e.seat].thinkTimeMs += (e.elapsedMs || 0);
  });

  if (dealStarts.length !== replay.totalDeals) {
    problems.push(`Expected ${replay.totalDeals} deal:start events, found ${dealStarts.length}`);
  }

  dealStarts.forEach(({ dealNumber, dealerSeat }) => {
    players.forEach(p => { p.bid = null; p.tricksWon = 0; });

    // Reconstruct the shuffle/deal exactly as the seeded RNG stream would
    // have produced it at this point in the match.
    const deck = new Deck(rng).generate().shuffle();
    const { hands, lastCard } = deck.deal(dealerSeat);
    const trumpSuit = lastCard.suit;

    // Replay bids (order doesn't affect legality here, just record them).
    (bidsByDeal[dealNumber] || []).forEach(({ seat, bid }) => {
      if (bid < 2 || bid > 13) problems.push(`Deal ${dealNumber}: seat ${seat} bid ${bid} outside [2,13]`);
      players[seat].bid = bid;
    });
    if ((bidsByDeal[dealNumber] || []).length !== 4) {
      problems.push(`Deal ${dealNumber}: expected 4 bids, found ${(bidsByDeal[dealNumber] || []).length}`);
    }

    // Replay tricks, validating legality of every recorded play.
    const plays = (playsByDeal[dealNumber] || []).slice()
      .sort((a, b) => a.trickNumber - b.trickNumber);
    let idx = 0;
    for (let trickNumber = 1; trickNumber <= 13; trickNumber++) {
      const trick = [];
      for (let i = 0; i < 4; i++) {
        const ev = plays[idx++];
        if (!ev || ev.trickNumber !== trickNumber) {
          problems.push(`Deal ${dealNumber} trick ${trickNumber}: missing a recorded play`);
          break;
        }
        const hand = hands[ev.seat];
        const legal = RuleEngine.getLegalCards(hand, trick, trumpSuit);
        const wasLegal = legal.some(c => c.suit === ev.card.suit && c.rank === ev.card.rank);
        if (!wasLegal) {
          problems.push(`Deal ${dealNumber} trick ${trickNumber}: seat ${ev.seat} played ` +
            `${cardId(ev.card)} which was ILLEGAL (legal set was {${legal.map(cardId).join(",")}})`);
        }
        hands[ev.seat] = hand.filter(c => !(c.suit === ev.card.suit && c.rank === ev.card.rank));
        trick.push({ player: ev.seat, card: ev.card });
      }
      if (trick.length === 4) {
        const winnerSeat = RuleEngine.determineTrickWinner(trick, trumpSuit);
        players[winnerSeat].tricksWon++;
      }
    }

    // Recompute and diff the deal score.
    const recorded = scoredByDeal[dealNumber];
    players.forEach(p => {
      const sc = RuleEngine.calculateDealScore(p.bid, p.tricksWon);
      p.score += sc;
      if (sc < 0) p.negativeDealCount++;
      const rec = recorded && recorded.scores.find(s => s.seat === p.seat);
      if (!rec) {
        problems.push(`Deal ${dealNumber}: no recorded score for seat ${p.seat}`);
      } else if (rec.tricksWon !== p.tricksWon || Math.abs(rec.scoreDelta - sc) > 1e-9) {
        problems.push(`Deal ${dealNumber} seat ${p.seat}: recomputed tricksWon=${p.tricksWon}` +
          ` scoreDelta=${sc}, but replay recorded tricksWon=${rec.tricksWon} scoreDelta=${rec.scoreDelta}`);
      }
    });
  });

  // Final scores / ranking.
  const ranked = players.slice().sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (a.negativeDealCount !== b.negativeDealCount) return a.negativeDealCount - b.negativeDealCount;
    return a.thinkTimeMs - b.thinkTimeMs;
  });
  const recomputedRanking = ranked.map(p => p.seat);

  players.forEach(p => {
    const rec = (replay.finalScores || []).find(f => f.seat === p.seat);
    if (!rec) { problems.push(`No recorded finalScores entry for seat ${p.seat}`); return; }
    if (Math.abs(rec.score - p.score) > 1e-9) {
      problems.push(`Seat ${p.seat}: recomputed final score ${p.score.toFixed(1)} != recorded ${rec.score.toFixed(1)}`);
    }
  });
  if (JSON.stringify(recomputedRanking) !== JSON.stringify(replay.ranking)) {
    problems.push(`Recomputed ranking [${recomputedRanking}] != recorded ranking [${replay.ranking}]`);
  }

  return { ok: problems.length === 0, problems, recomputedScores: players.map(p => p.score), recomputedRanking };
}

if (require.main === module) {
  const file = process.argv[2];
  if (!file) { console.error("Usage: node replay-verify.js <replay.json>"); process.exit(1); }
  const replay = JSON.parse(fs.readFileSync(file, "utf8"));

  console.log(`\n=== Verifying replay: ${file} ===`);
  console.log(`  matchSeed: ${replay.matchSeed}   rulesetVersion: ${replay.rulesetVersion}   build: ${replay.build}\n`);

  const result = verifyReplay(replay);
  if (result.ok) {
    console.log("  ✓ Fully reproduced. Every move legal, every score matches, ranking matches.");
  } else {
    console.log(`  ✗ ${result.problems.length} problem(s) found:`);
    result.problems.forEach(p => console.log(`    - ${p}`));
  }
  console.log("");
  process.exit(result.ok ? 0 : 1);
}

module.exports = { verifyReplay };
