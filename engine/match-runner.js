/* ============================================================================
   match-runner.js
   Generalized version of simulate-match.js's core loop: runs one 8-deal
   match using a per-seat pluggable strategy (see bot-strategies.js) instead
   of the single hardcoded "lowest legal card" bot.

   Produces:
     replay — same shape simulate-match.js writes, fully compatible with
              replay-verify.js (extra fields like seatStrategies are
              additive and simply ignored by the verifier).
     stats  — extra analytics for batch comparison (bid accuracy, ruffs,
              tricks trumped, per-seat outcome by strategy).

   rule-engine.js / deck-engine.js are used exactly as shipped — this file
   only swaps out WHICH card/bid a bot chooses, never what's legal.
============================================================================ */
const { RuleEngine } = require('./rule-engine.js');
const { Deck, TurnManager } = require('./deck-engine.js');
const { mulberry32, seedToLabel } = require('./seeded-rng.js');

const CONFIG = {
  RULESET_VERSION: '2.0.0',
  REPLAY_FORMAT_VERSION: '1.0.0',
  GAME_VERSION: '0.3.0',
  BUILD: 'Build 03 (sim)',
  TOTAL_DEALS: 8,
  MIN_BID: 2,
  MAX_BID: 13,
};

function runMatch(seed, seatStrategies){
  const rng = mulberry32(seed);
  const events = [];
  const players = [0,1,2,3].map(seat=>({
    seat, bid:null, tricksWon:0, score:0, negativeDealCount:0, thinkTimeMs:0,
    exactBids:0, dealsPlayed:0, bidError:0
  }));

  const dealStats = [];
  let dealerSeat = 0;

  for(let dealNumber=1; dealNumber<=CONFIG.TOTAL_DEALS; dealNumber++){
    players.forEach(p=>{ p.bid=null; p.tricksWon=0; });

    const deck = new Deck(rng).generate().shuffle();
    const { hands, lastCard } = deck.deal(dealerSeat);
    const trumpSuit = lastCard.suit;

    events.push({ type:'deal:start', dealNumber, dealerSeat });

    const bidOrder = TurnManager.getBidOrder(dealerSeat);
    bidOrder.forEach(seat=>{
      const bid = seatStrategies[seat].bidEstimate(hands[seat], trumpSuit);
      players[seat].bid = bid;
      events.push({ type:'bid:placed', dealNumber, seat, bid, elapsedMs:0 });
    });

    let leaderSeat = TurnManager.nextSeat(dealerSeat);
    let dealTricksTrumped = 0, dealRuffs = 0;

    for(let trickNumber=1; trickNumber<=13; trickNumber++){
      const playOrder = TurnManager.getPlayOrder(leaderSeat);
      const trick = [];
      playOrder.forEach(seat=>{
        const hand = hands[seat];
        const legal = RuleEngine.getLegalCards(hand, trick, trumpSuit);
        const chosen = seatStrategies[seat].choosePlay(hand, trick, trumpSuit, legal);
        const ledSuit = trick.length ? trick[0].card.suit : chosen.suit;
        const isRuff = trick.length>0 && chosen.suit===trumpSuit && ledSuit!==trumpSuit;
        if(isRuff) dealRuffs++;
        hands[seat] = hand.filter(c=>!(c.suit===chosen.suit && c.rank===chosen.rank));
        trick.push({ player:seat, card:chosen });
        events.push({ type:'card:played', dealNumber, trickNumber, seat, card:chosen, elapsedMs:0 });
      });
      if(RuleEngine.trumpPlayed(trick, trumpSuit)) dealTricksTrumped++;
      const winnerSeat = RuleEngine.determineTrickWinner(trick, trumpSuit);
      players[winnerSeat].tricksWon++;
      leaderSeat = winnerSeat;
    }

    const dealScores = players.map(p=>{
      const sc = RuleEngine.calculateDealScore(p.bid, p.tricksWon);
      p.score += sc;
      if(sc<0) p.negativeDealCount++;
      p.dealsPlayed++;
      p.bidError += Math.abs(p.tricksWon - p.bid);
      if(p.tricksWon===p.bid) p.exactBids++;
      return { seat:p.seat, bid:p.bid, tricksWon:p.tricksWon, scoreDelta:sc };
    });
    events.push({ type:'deal:scored', dealNumber, scores:dealScores });
    dealStats.push({ dealNumber, trumpSuit, tricksTrumped:dealTricksTrumped, ruffs:dealRuffs });

    dealerSeat = TurnManager.rotateDealer(dealerSeat);
  }

  const ranked = players.slice().sort((a,b)=>{
    if(b.score!==a.score) return b.score-a.score;
    if(a.negativeDealCount!==b.negativeDealCount) return a.negativeDealCount-b.negativeDealCount;
    return a.thinkTimeMs-b.thinkTimeMs;
  });

  const finalScores = players.map(p=>({ seat:p.seat, score:p.score, negativeDealCount:p.negativeDealCount, thinkTimeMs:p.thinkTimeMs }));
  const ranking = ranked.map(p=>p.seat);
  events.push({ type:'game:end', finalScores, ranking });

  const replay = {
    replayFormatVersion: CONFIG.REPLAY_FORMAT_VERSION,
    rulesetVersion: CONFIG.RULESET_VERSION,
    gameVersion: CONFIG.GAME_VERSION,
    build: CONFIG.BUILD,
    matchSeed: seedToLabel(seed),
    seatStrategies: seatStrategies.map(s=>s.name),
    totalDeals: CONFIG.TOTAL_DEALS,
    generatedAt: new Date().toISOString(),
    events,
    finalScores,
    ranking
  };

  const stats = {
    matchSeed: replay.matchSeed,
    seatStrategies: replay.seatStrategies,
    perSeat: players.map(p=>({
      seat: p.seat,
      strategy: seatStrategies[p.seat].name,
      finalScore: p.score,
      negativeDeals: p.negativeDealCount,
      exactBids: p.exactBids,
      dealsPlayed: p.dealsPlayed,
      meanAbsBidError: p.bidError/p.dealsPlayed,
      rank: ranking.indexOf(p.seat)+1
    })),
    dealStats
  };

  return { replay, stats };
}

module.exports = { runMatch, CONFIG };
