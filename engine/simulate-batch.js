/* ============================================================================
   simulate-batch.js
   Runs N matches with a chosen strategy assignment across the four seats and
   prints aggregate stats comparing strategies head-to-head. Also writes one
   sample replay (match #1) so it can be independently spot-checked with
   replay-verify.js — proving the batch harness didn't introduce any illegal
   moves relative to the shipped RuleEngine.

   Usage:
     node simulate-batch.js [numMatches] [assignment] [sampleOutFile]

   assignment: comma-separated strategy per seat 0,1,2,3.
     "naive,naive,heuristic,heuristic"   (default — direct head-to-head)
     "naive,naive,naive,naive"           (baseline: current shipped bots)
     "heuristic,heuristic,heuristic,heuristic"

   Available strategies: naive, heuristic  (see bot-strategies.js)
============================================================================ */
const fs = require('fs');
const { runMatch } = require('./match-runner.js');
const { generateSeed } = require('./seeded-rng.js');
const { naiveStrategy, heuristicStrategy, hardStrategy, easyStrategy, mediumStrategy } = require('./bot-strategies.js');

const STRATS = {
  naive: naiveStrategy, heuristic: heuristicStrategy, hard: hardStrategy,
  easy: easyStrategy, medium: mediumStrategy
};

function parseAssignment(str){
  return str.split(',').map(s=>{
    const strat = STRATS[s.trim()];
    if(!strat) throw new Error(`Unknown strategy "${s}". Options: ${Object.keys(STRATS).join(', ')}`);
    return strat;
  });
}

function runBatch(numMatches, seatStrategies){
  const nameOf = seatStrategies.map(s=>s.name);
  const byStrategy = {};
  Object.values(STRATS).forEach(s=>{
    byStrategy[s.name] = {
      games: 0, wins: 0, totalScore: 0, totalNegDeals: 0,
      totalExactBids: 0, totalDealsPlayed: 0, totalAbsBidError: 0
    };
  });

  let totalTricksTrumped = 0, totalTricks = 0, totalRuffs = 0;
  let firstReplay = null;

  for(let i=0;i<numMatches;i++){
    const seed = generateSeed();
    const { replay, stats } = runMatch(seed, seatStrategies);
    if(i===0) firstReplay = replay;

    stats.perSeat.forEach(p=>{
      const acc = byStrategy[p.strategy];
      acc.games++;
      if(p.rank===1) acc.wins++;
      acc.totalScore += p.finalScore;
      acc.totalNegDeals += p.negativeDeals;
      acc.totalExactBids += p.exactBids;
      acc.totalDealsPlayed += p.dealsPlayed;
      acc.totalAbsBidError += p.meanAbsBidError * p.dealsPlayed;
    });

    stats.dealStats.forEach(d=>{
      totalTricks += 13;
      totalTricksTrumped += d.tricksTrumped;
      totalRuffs += d.ruffs;
    });
  }

  const summary = {};
  Object.entries(byStrategy).forEach(([name, acc])=>{
    if(acc.games===0) return;
    summary[name] = {
      seatsUsingThisStrategy: nameOf.filter(n=>n===name).length,
      gamesPlayed: acc.games,
      winRate: +(acc.wins/acc.games*100).toFixed(1),
      avgFinalScore: +(acc.totalScore/acc.games).toFixed(2),
      avgNegativeDealsPerGame: +(acc.totalNegDeals/acc.games).toFixed(2),
      exactBidRate: +(acc.totalExactBids/acc.totalDealsPlayed*100).toFixed(1),
      meanAbsBidError: +(acc.totalAbsBidError/acc.totalDealsPlayed).toFixed(2)
    };
  });

  return {
    numMatches,
    seatAssignment: nameOf,
    summary,
    tricksTrumpedRate: +(totalTricksTrumped/totalTricks*100).toFixed(1),
    ruffsPerMatch: +(totalRuffs/numMatches).toFixed(2),
    firstReplay
  };
}

if(require.main===module){
  const numMatches = parseInt(process.argv[2],10) || 200;
  const assignmentStr = process.argv[3] || 'naive,naive,heuristic,heuristic';
  const sampleOutFile = process.argv[4] || 'batch_sample_replay.json';

  const seatStrategies = parseAssignment(assignmentStr);
  console.log(`\nRunning ${numMatches} matches — seats: ${seatStrategies.map(s=>s.name).join(', ')}\n`);

  const result = runBatch(numMatches, seatStrategies);

  fs.writeFileSync(sampleOutFile, JSON.stringify(result.firstReplay, null, 2));
  delete result.firstReplay;

  console.log('=== Per-strategy results ===');
  Object.entries(result.summary).forEach(([name, s])=>{
    console.log(`\n  ${name}  (${s.seatsUsingThisStrategy} seat(s), ${s.gamesPlayed} games)`);
    console.log(`    Win rate:            ${s.winRate}%`);
    console.log(`    Avg final score:     ${s.avgFinalScore}`);
    console.log(`    Avg negative deals:  ${s.avgNegativeDealsPerGame} / 8`);
    console.log(`    Exact-bid rate:      ${s.exactBidRate}%`);
    console.log(`    Mean |bid - won|:    ${s.meanAbsBidError}`);
  });
  console.log(`\n  Tricks trumped:  ${result.tricksTrumpedRate}%`);
  console.log(`  Ruffs / match:   ${result.ruffsPerMatch}`);
  console.log(`\n  Sample replay (match 1) -> ${sampleOutFile}  (verify with: node replay-verify.js ${sampleOutFile})\n`);
}

module.exports = { runBatch, parseAssignment };
