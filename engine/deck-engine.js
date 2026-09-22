/* ============================================================================
   deck-engine.js
   Node-side mirror of the CardFactory / Deck / TurnManager classes embedded
   in TrumpMaster_v0_2_Build03.html. Kept logically identical so a match
   simulated/verified here matches what the browser build would produce for
   the same seed.
============================================================================ */
const SUITS = ["S", "H", "D", "C"];

class CardFactory {
  static createDeck() {
    const deck = [];
    for (const s of SUITS) {
      for (let r = 2; r <= 14; r++) {
        deck.push(Object.freeze({ suit: s, rank: r, id: `${s}${r}` }));
      }
    }
    return Object.freeze(deck);
  }
}

class Deck {
  constructor(rng) {
    this.cards = [];
    this.rng = rng || Math.random; // same fallback as the browser build
  }
  generate() { this.cards = CardFactory.createDeck().slice(); return this; }
  shuffle() {
    const a = this.cards;
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return this;
  }
  // Deals starting from seat (dealerSeat+1), round-robin, so the dealer
  // receives the 52nd (last) card.
  deal(dealerSeat) {
    const hands = [[], [], [], []];
    const order = [];
    for (let i = 0; i < 52; i++) order.push((dealerSeat + 1 + i) % 4);
    for (let i = 0; i < 52; i++) hands[order[i]].push(this.cards[i]);
    for (let p = 0; p < 4; p++) hands[p].sort((a, b) => a.suit.localeCompare(b.suit) || a.rank - b.rank);
    const lastCard = this.cards[51];
    return { hands, lastCard };
  }
  reset() { this.cards = []; return this; }
}

class TurnManager {
  static nextSeat(seat) { return (seat + 1) % 4; }
  static rotateDealer(dealerSeat) { return TurnManager.nextSeat(dealerSeat); }
  static getBidOrder(dealerSeat) { return [1, 2, 3, 0].map(off => (dealerSeat + off) % 4); }
  static getPlayOrder(leaderSeat) { return [0, 1, 2, 3].map(off => (leaderSeat + off) % 4); }
}

module.exports = { CardFactory, Deck, TurnManager, SUITS };
