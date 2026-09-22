/* ============================================================================
   rule-engine.js
   VERBATIM extraction of the RuleEngine class from
   TrumpMaster_v0_2_Build02.html (lines ~413-508), so the verification suite
   tests the REAL shipped logic, not a re-implementation.
   Card shape: { suit:'S'|'H'|'D'|'C', rank:2..14 }  (14=A,13=K,12=Q,11=J)
   Trick shape: [{ player:0..3, card:{suit,rank} }, ...]
============================================================================ */
class RuleEngine{
  static ledSuit(trick){ return trick.length ? trick[0].card.suit : null; }

  static highestOfSuit(trick, suit){
    let best = null;
    trick.forEach(play=>{ if(play.card.suit===suit && (!best || play.card.rank>best.card.rank)) best = play; });
    return best;
  }

  static trumpPlayed(trick, trumpSuit){ return trick.some(p=>p.card.suit===trumpSuit); }

  static getLegalCards(hand, trick, trumpSuit){
    const led = RuleEngine.ledSuit(trick);
    if(led===null) return hand.slice();
    const ledCards = hand.filter(c=>c.suit===led);
    if(ledCards.length>0){
      if(led===trumpSuit){
        const bestTrump = RuleEngine.highestOfSuit(trick, trumpSuit);
        const higherTrump = ledCards.filter(c=>c.rank>bestTrump.card.rank);
        if(higherTrump.length>0) return higherTrump;
        return ledCards;
      }
      if(RuleEngine.trumpPlayed(trick, trumpSuit))
        return ledCards;
      const bestLed = RuleEngine.highestOfSuit(trick, led);
      const higherLed = ledCards.filter(c=>c.rank>bestLed.card.rank);
      if(higherLed.length>0) return higherLed;
      return ledCards;
    }
    const trumpCards = hand.filter(c=>c.suit===trumpSuit);
    if(trumpCards.length===0) return hand.slice();
    if(led!==trumpSuit && !RuleEngine.trumpPlayed(trick,trumpSuit))
      return trumpCards;
    if(led!==trumpSuit){
      const bestTrump=RuleEngine.highestOfSuit(trick,trumpSuit);
      const higherTrump=trumpCards.filter(c=>c.rank>bestTrump.card.rank);
      if(higherTrump.length>0) return higherTrump;
    }
    return hand.slice();
  }

  static validateMove(hand, trick, trumpSuit, card){
    return RuleEngine.getLegalCards(hand, trick, trumpSuit).some(c=>c.suit===card.suit && c.rank===card.rank);
  }

  static determineTrickWinner(trick, trumpSuit){
    const led = RuleEngine.ledSuit(trick);
    if(RuleEngine.trumpPlayed(trick, trumpSuit)) return RuleEngine.highestOfSuit(trick, trumpSuit).player;
    return RuleEngine.highestOfSuit(trick, led).player;
  }

  static calculateDealScore(bid, won){
    if(won===bid) return bid;
    if(won<bid) return -bid;
    if(won<2*bid) return bid + (won-bid)/10;
    return -bid;
  }

  static calculateFinalRanking(players){
    return players.slice().sort((a,b)=>{
      if(b.score!==a.score) return b.score-a.score;
      if(a.negativeDealCount!==b.negativeDealCount) return a.negativeDealCount-b.negativeDealCount;
      return a.thinkTimeMs-b.thinkTimeMs;
    });
  }
}
module.exports = { RuleEngine };
