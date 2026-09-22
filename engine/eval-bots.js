/* ============================================================================
   eval-bots.js
   Head-to-head evaluation of bot strategies with confidence intervals.

   simulate-batch.js compares strategies with fixed seats, so a strategy's result
   mixes in seat effects (dealer position, bid order). Here every match seed is
   played once per rotation of the strategy line-up (4 rotations), so each
   strategy sits in every seat on exactly the same cards. Reported per strategy:
   mean match score (8 deals) +/- 95% CI, win rate (rank 1 by the official
   tie-break), exact-bid rate and negative-deal rate.

   Usage:  node eval-bots.js [numSeeds=500] [lineup=hard,medium,easy,easy] [firstSeed=1000]
============================================================================ */
const { runMatch } = require('./match-runner.js');
const S = require('./bot-strategies.js');

const STRATS = { naive: S.naiveStrategy, heuristic: S.heuristicStrategy, hard: S.hardStrategy,
                 easy: S.easyStrategy, medium: S.mediumStrategy };

const numSeeds = parseInt(process.argv[2] || '500', 10);
const lineup = (process.argv[3] || 'hard,medium,easy,easy').split(',').map(s => s.trim());
const firstSeed = parseInt(process.argv[4] || '1000', 10);
lineup.forEach(n => { if (!STRATS[n]) throw new Error(`unknown strategy ${n}`); });


const acc = {};
lineup.forEach(n => { acc[n] = acc[n] || { scores: [], wins: 0, seats: 0, exact: 0, deals: 0, neg: 0 }; });

for (let k = 0; k < numSeeds; k++) {
  const seed = (firstSeed + k) >>> 0;
  for (let rot = 0; rot < 4; rot++) {
    const names = [0, 1, 2, 3].map(seat => lineup[(seat + rot) % 4]);
    // the engine's own ranking applies the official tie-break (score, negative deals, think time)
    const { stats } = runMatch(seed, names.map(n => STRATS[n]));
    stats.perSeat.forEach(p => {
      const a = acc[names[p.seat]];
      a.scores.push(p.finalScore); a.seats++; if (p.rank === 1) a.wins++;
      a.exact += p.exactBids; a.deals += p.dealsPlayed; a.neg += p.negativeDeals;
    });
  }
}

function meanCI(x) {
  const n = x.length, m = x.reduce((a, b) => a + b, 0) / n;
  const v = x.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1);
  return [m, 1.96 * Math.sqrt(v / n)];
}
const out = { numSeeds, matchesPlayed: numSeeds * 4, lineup, strategies: {} };
for (const [n, a] of Object.entries(acc)) {
  const [m, h] = meanCI(a.scores);
  // lineups can repeat a strategy; its "fair" win rate is copies/4
  const copies = lineup.filter(x => x === n).length;
  out.strategies[n] = { meanMatchScore: +m.toFixed(3), ci95: +h.toFixed(3),
    winRate: +(a.wins / (numSeeds * 4)).toFixed(3), fairWinRate: copies / 4,
    exactBidRate: +(a.exact / a.deals).toFixed(3), negativeDealRate: +(a.neg / a.deals).toFixed(3) };
}
console.log(JSON.stringify(out, null, 2));
