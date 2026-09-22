/* ============================================================================
   simulate-match.js
   Runs one full 8-deal TrumpMaster match headlessly (all 4 seats bot-driven,
   same heuristics as the HTML build) using the real RuleEngine and a seeded
   Deck, and writes out a replay file.

   Usage:
     node simulate-match.js [seedHexOrDecimal] [outFile]

   If no seed is given, a fresh random one is generated and printed — copy it
   down if you want to reproduce this exact match later.
============================================================================ */
const fs = require("fs");
const { RuleEngine } = require("./rule-engine.js");
const { Deck, TurnManager } = require("./deck-engine.js");
const { mulberry32, generateSeed, seedToLabel, labelToSeed } = require("./seeded-rng.js");

const CONFIG = {
  RULESET_VERSION: "2.0.0",
  REPLAY_FORMAT_VERSION: "1.0.0",
  GAME_VERSION: "0.3.0",
  BUILD: "Build 03",
  TOTAL_DEALS: 8,
  MIN_BID: 2,
  MAX_BID: 13,
};

function botBidEstimate(hand, trumpSuit) {
  let est = 0;
  hand.forEach(c => {
    if (c.suit === trumpSuit) est += 0.55;
    if (c.rank === 14) est += 0.8; else if (c.rank === 13) est += 0.55; else if (c.rank === 12) est += 0.3;
  });
  let bid = Math.round(est);
  if (bid < CONFIG.MIN_BID) bid = CONFIG.MIN_BID;
  if (bid > CONFIG.MAX_BID) bid = CONFIG.MAX_BID;
  return bid;
}

function botChoosePlay(legalCards) {
  return legalCards.slice().sort((a, b) => a.rank - b.rank)[0];
}

function simulateMatch(seed) {
  const rng = mulberry32(seed);
  const events = [];
  const players = [0, 1, 2, 3].map(seat => ({
    seat, bid: null, tricksWon: 0, score: 0, negativeDealCount: 0, thinkTimeMs: 0
  }));

  let dealerSeat = 0;

  for (let dealNumber = 1; dealNumber <= CONFIG.TOTAL_DEALS; dealNumber++) {
    players.forEach(p => { p.bid = null; p.tricksWon = 0; });

    const deck = new Deck(rng).generate().shuffle();
    const { hands, lastCard } = deck.deal(dealerSeat);
    const trumpSuit = lastCard.suit;

    events.push({ type: "deal:start", dealNumber, dealerSeat });

    // --- Bidding ---
    const bidOrder = TurnManager.getBidOrder(dealerSeat);
    bidOrder.forEach(seat => {
      const bid = botBidEstimate(hands[seat], trumpSuit);
      players[seat].bid = bid;
      events.push({ type: "bid:placed", dealNumber, seat, bid, elapsedMs: 0 });
    });

    // --- Playing 13 tricks ---
    let leaderSeat = TurnManager.nextSeat(dealerSeat);
    for (let trickNumber = 1; trickNumber <= 13; trickNumber++) {
      const playOrder = TurnManager.getPlayOrder(leaderSeat);
      const trick = [];
      playOrder.forEach(seat => {
        const legal = RuleEngine.getLegalCards(hands[seat], trick, trumpSuit);
        const chosen = botChoosePlay(legal);
        hands[seat] = hands[seat].filter(c => !(c.suit === chosen.suit && c.rank === chosen.rank));
        trick.push({ player: seat, card: chosen });
        events.push({ type: "card:played", dealNumber, trickNumber, seat, card: chosen, elapsedMs: 0 });
      });
      const winnerSeat = RuleEngine.determineTrickWinner(trick, trumpSuit);
      players[winnerSeat].tricksWon++;
      leaderSeat = winnerSeat;
    }

    // --- Score the deal ---
    const dealScores = players.map(p => {
      const sc = RuleEngine.calculateDealScore(p.bid, p.tricksWon);
      p.score += sc;
      if (sc < 0) p.negativeDealCount++;
      return { seat: p.seat, bid: p.bid, tricksWon: p.tricksWon, scoreDelta: sc };
    });
    events.push({ type: "deal:scored", dealNumber, scores: dealScores });

    dealerSeat = TurnManager.rotateDealer(dealerSeat);
  }

  const ranked = players.slice().sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (a.negativeDealCount !== b.negativeDealCount) return a.negativeDealCount - b.negativeDealCount;
    return a.thinkTimeMs - b.thinkTimeMs;
  });

  const finalScores = players.map(p => ({
    seat: p.seat, score: p.score, negativeDealCount: p.negativeDealCount, thinkTimeMs: p.thinkTimeMs
  }));
  const ranking = ranked.map(p => p.seat);

  events.push({ type: "game:end", finalScores, ranking });

  return {
    replayFormatVersion: CONFIG.REPLAY_FORMAT_VERSION,
    rulesetVersion: CONFIG.RULESET_VERSION,
    gameVersion: CONFIG.GAME_VERSION,
    build: CONFIG.BUILD,
    matchSeed: seedToLabel(seed),
    totalDeals: CONFIG.TOTAL_DEALS,
    generatedAt: new Date().toISOString(),
    events,
    finalScores,
    ranking
  };
}

// --- CLI ---
if (require.main === module) {
  const seedArg = process.argv[2];
  const outFile = process.argv[3] || "replay.json";
  const seed = seedArg ? (/^[0-9a-fA-F]{1,8}$/.test(seedArg) && seedArg.length === 8 ? labelToSeed(seedArg) : parseInt(seedArg, 10)) : generateSeed();

  const replay = simulateMatch(seed >>> 0);
  fs.writeFileSync(outFile, JSON.stringify(replay, null, 2));
  console.log(`Match simulated with seed ${replay.matchSeed} -> ${outFile}`);
  console.log(`Final ranking (seat order, 1st first): ${replay.ranking.join(", ")}`);
  console.log(`Scores: ${replay.finalScores.map(p => `seat${p.seat}=${p.score.toFixed(1)}`).join("  ")}`);
}

module.exports = { simulateMatch };
