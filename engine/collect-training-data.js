/* ============================================================================
   collect-training-data.js
   Regenerates the (hand features -> tricks actually won) dataset used to fit
   heuristicStrategy's bid regression. Bidding does not affect trick outcomes
   in this engine (only choosePlay does), so we can isolate "what is this
   hand worth under heuristic play" cleanly: play out full matches with
   heuristicStrategy.choosePlay on all four seats, and for every hand dealt,
   record its shape features against the tricks that seat actually won that
   deal.

   Usage:
     node collect-training-data.js [numMatches] [outFile]

   Writes a CSV: trumpCount,aceCount,kingCount,queenCount,voidCount,
   singletonCount,tricksWon
============================================================================ */
const fs = require('fs');
const { RuleEngine } = require('./rule-engine.js');
const { Deck, TurnManager } = require('./deck-engine.js');
const { mulberry32, generateSeed } = require('./seeded-rng.js');
const { heuristicStrategy } = require('./bot-strategies.js');

const TOTAL_DEALS = 8;

function groupBySuit(hand) {
  const g = { S: [], H: [], D: [], C: [] };
  hand.forEach(c => g[c.suit].push(c));
  return g;
}

function featuresOf(hand, trumpSuit) {
  const bySuit = groupBySuit(hand);
  const trumpCount = bySuit[trumpSuit].length;
  let aceCount = 0, kingCount = 0, queenCount = 0;
  hand.forEach(c => {
    if (c.rank === 14) aceCount++; else if (c.rank === 13) kingCount++; else if (c.rank === 12) queenCount++;
  });
  let voidCount = 0, singletonCount = 0;
  ['S', 'H', 'D', 'C'].forEach(s => {
    if (s === trumpSuit) return;
    if (bySuit[s].length === 0) voidCount++;
    else if (bySuit[s].length === 1) singletonCount++;
  });
  return { trumpCount, aceCount, kingCount, queenCount, voidCount, singletonCount };
}

function collect(numMatches) {
  const rows = [];

  for (let m = 0; m < numMatches; m++) {
    const seed = generateSeed();
    const rng = mulberry32(seed);
    let dealerSeat = 0;

    for (let dealNumber = 1; dealNumber <= TOTAL_DEALS; dealNumber++) {
      const deck = new Deck(rng).generate().shuffle();
      const { hands, lastCard } = deck.deal(dealerSeat);
      const trumpSuit = lastCard.suit;

      // Snapshot pre-play features for all four hands.
      const dealFeatures = hands.map(h => featuresOf(h, trumpSuit));
      const tricksWon = [0, 0, 0, 0];

      // Play out the deal with heuristicStrategy on all four seats — bid
      // doesn't factor into play at all in this engine.
      let leaderSeat = TurnManager.nextSeat(dealerSeat);
      for (let trickNumber = 1; trickNumber <= 13; trickNumber++) {
        const playOrder = TurnManager.getPlayOrder(leaderSeat);
        const trick = [];
        playOrder.forEach(seat => {
          const legal = RuleEngine.getLegalCards(hands[seat], trick, trumpSuit);
          const chosen = heuristicStrategy.choosePlay(hands[seat], trick, trumpSuit, legal);
          hands[seat] = hands[seat].filter(c => !(c.suit === chosen.suit && c.rank === chosen.rank));
          trick.push({ player: seat, card: chosen });
        });
        const winnerSeat = RuleEngine.determineTrickWinner(trick, trumpSuit);
        tricksWon[winnerSeat]++;
        leaderSeat = winnerSeat;
      }

      for (let seat = 0; seat < 4; seat++) {
        rows.push({ ...dealFeatures[seat], tricksWon: tricksWon[seat] });
      }

      dealerSeat = TurnManager.rotateDealer(dealerSeat);
    }
  }

  return rows;
}

if (require.main === module) {
  const numMatches = parseInt(process.argv[2], 10) || 1500;
  const outFile = process.argv[3] || 'training-data.csv';

  console.log(`Simulating ${numMatches} matches (${numMatches * 8 * 4} hand samples) under heuristic play...`);
  const rows = collect(numMatches);

  const header = 'trumpCount,aceCount,kingCount,queenCount,voidCount,singletonCount,tricksWon';
  const lines = rows.map(r => `${r.trumpCount},${r.aceCount},${r.kingCount},${r.queenCount},${r.voidCount},${r.singletonCount},${r.tricksWon}`);
  fs.writeFileSync(outFile, [header, ...lines].join('\n'));

  console.log(`Wrote ${rows.length} rows -> ${outFile}`);
  console.log('\nFirst 15 rows:');
  console.log(header);
  lines.slice(0, 15).forEach(l => console.log(l));
}

module.exports = { collect, featuresOf };
