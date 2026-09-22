/* ============================================================================
   train-rl-bidder.js
   Trains the "hard" bot's bidder with reinforcement learning instead of
   regression.

   Why RL here, and how it differs from the OLS bidder:
     The regression bidder (heuristicStrategy / "medium") learns to predict
     E[tricks won | hand], then rounds that to a bid. That's a reasonable
     proxy, but it's not actually what we want to maximize — the real
     scoring rule (RuleEngine.calculateDealScore) is asymmetric: underbidding
     is punished much harder than overbidding by the same margin, and an
     exact bid is worth strictly more than "close". A predictor trained to
     minimize squared tricks-error has no idea any of that asymmetry exists.

   The RL bidder learns directly against the real reward (deal score),
   deal-by-deal, using REINFORCE (a policy-gradient method):
     - Policy: bid ~ Normal(mean = w . features, sigma), sigma anneals down
       as training progresses (explore early, exploit late).
     - After each deal, compute reward = calculateDealScore(bid, tricksWon).
     - Update: w += lr * (reward - baseline) * (bid - mean)/sigma^2 * features
       (baseline = running average reward, reduces variance without biasing
       the gradient — standard REINFORCE-with-baseline).
     - Play (choosePlay) is held fixed to heuristicStrategy's play throughout
       training, so — same isolation trick as the OLS pipeline — the only
       thing being learned is "how should I bid given this hand", decoupled
       from card-play skill.

   Usage:
     node train-rl-bidder.js [numMatches]
   Writes the learned weights into rl-bid-weights.json and prints a reward
   curve (avg deal score per 500-match chunk) so you can see it improving.
============================================================================ */
const fs = require('fs');
const { RuleEngine } = require('./rule-engine.js');
const { Deck, TurnManager } = require('./deck-engine.js');
const { mulberry32, generateSeed } = require('./seeded-rng.js');
const { heuristicStrategy } = require('./bot-strategies.js');
const { featuresOf } = require('./collect-training-data.js');

const TOTAL_DEALS = 8;
const FEATURE_NAMES = ['trumpCount', 'aceCount', 'kingCount', 'queenCount', 'voidCount', 'singletonCount'];
const LR = 0.0002;
const SIGMA_START = 1.5;
const SIGMA_END = 0.6;
const ADVANTAGE_CLIP = 2;      // reward normalization keeps gradient steps bounded
const GRAD_CLIP = 1;           // clip the per-weight update itself as a second safety net
const WEIGHT_CLIP = 3;         // hard bound on any single coefficient (OLS coefficients
                                // all landed in [-0.3, 0.7], so this is generous headroom
                                // while still catching runaway drift over 100k+ updates)

function clampBid(n) { return Math.max(2, Math.min(13, n)); }

function featureVec(f) { return [1, f.trumpCount, f.aceCount, f.kingCount, f.queenCount, f.voidCount, f.singletonCount]; }

function dot(w, x) { return w.reduce((s, wi, i) => s + wi * x[i], 0); }

function trainRL(numMatches) {
  // Warm-start from the OLS-fitted heuristic weights rather than zeros —
  // the OLS bidder is already a decent tricks-estimator; RL only needs to
  // learn the *adjustment* for the asymmetric scoring rule, which is a much
  // shorter path than learning bidding from scratch.
  const c = heuristicStrategy.__coefficients || {
    bias: -0.293, trumpCount: 0.674, aceCount: 0.543, kingCount: 0.404,
    queenCount: 0.290, voidCount: 0.561, singletonCount: 0.398
  };
  let w = [c.bias, c.trumpCount, c.aceCount, c.kingCount, c.queenCount, c.voidCount, c.singletonCount];

  // Running reward mean/std (Welford-style EMA) so the advantage fed into
  // the gradient is normalized — without this, raw deal-score rewards
  // (which range roughly -13..+14) blow the weights up within a few hundred
  // updates, since REINFORCE's gradient scales linearly with the reward.
  let baseline = 0, rewardVar = 4;
  const BASELINE_DECAY = 0.99;
  const chunkSize = 500;
  const curve = [];
  let chunkRewardSum = 0, chunkDeals = 0;

  for (let m = 0; m < numMatches; m++) {
    const sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * Math.min(1, m / (numMatches * 0.7));
    const seed = generateSeed();
    const rng = mulberry32(seed);
    let dealerSeat = 0;

    for (let dealNumber = 1; dealNumber <= TOTAL_DEALS; dealNumber++) {
      const deck = new Deck(rng).generate().shuffle();
      const { hands, lastCard } = deck.deal(dealerSeat);
      const trumpSuit = lastCard.suit;

      // Only seat 0's bid is under the learning policy; seats 1-3 also use
      // it (all four seats share the same weights, all training in
      // parallel each deal — 4x the learning signal per match).
      const bids = [], means = [], xs = [];
      for (let seat = 0; seat < 4; seat++) {
        const f = featuresOf(hands[seat], trumpSuit);
        const x = featureVec(f);
        const mean = dot(w, x);
        // Gaussian exploration noise, sampled via Box-Muller from rng().
        const u1 = Math.max(rng(), 1e-9), u2 = rng();
        const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        const rawBid = mean + sigma * noise;
        bids.push(clampBid(Math.round(rawBid)));
        means.push(mean);
        xs.push(x);
      }

      // Play out the deal with fixed heuristic play.
      const handsCopy = hands.map(h => h.slice());
      let leaderSeat = TurnManager.nextSeat(dealerSeat);
      const tricksWon = [0, 0, 0, 0];
      for (let t = 1; t <= 13; t++) {
        const playOrder = TurnManager.getPlayOrder(leaderSeat);
        const trick = [];
        playOrder.forEach(seat => {
          const legal = RuleEngine.getLegalCards(handsCopy[seat], trick, trumpSuit);
          const chosen = heuristicStrategy.choosePlay(handsCopy[seat], trick, trumpSuit, legal);
          handsCopy[seat] = handsCopy[seat].filter(c2 => !(c2.suit === chosen.suit && c2.rank === chosen.rank));
          trick.push({ player: seat, card: chosen });
        });
        const winnerSeat = RuleEngine.determineTrickWinner(trick, trumpSuit);
        tricksWon[winnerSeat]++;
        leaderSeat = winnerSeat;
      }

      // REINFORCE update, one gradient step per seat's bid this deal.
      for (let seat = 0; seat < 4; seat++) {
        const reward = RuleEngine.calculateDealScore(bids[seat], tricksWon[seat]);
        const rawAdvantage = reward - baseline;
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward;
        rewardVar = BASELINE_DECAY * rewardVar + (1 - BASELINE_DECAY) * rawAdvantage * rawAdvantage;
        const advantage = Math.max(-ADVANTAGE_CLIP, Math.min(ADVANTAGE_CLIP, rawAdvantage / Math.sqrt(rewardVar + 1e-6)));
        const grad = (bids[seat] - means[seat]) / (sigma * sigma);
        for (let i = 0; i < w.length; i++) {
          const step = Math.max(-GRAD_CLIP, Math.min(GRAD_CLIP, LR * advantage * grad * xs[seat][i]));
          w[i] = Math.max(-WEIGHT_CLIP, Math.min(WEIGHT_CLIP, w[i] + step));
        }

        chunkRewardSum += reward;
        chunkDeals++;
      }

      dealerSeat = TurnManager.rotateDealer(dealerSeat);
    }

    if ((m + 1) % chunkSize === 0 || m === numMatches - 1) {
      curve.push({ throughMatch: m + 1, avgDealScore: +(chunkRewardSum / chunkDeals).toFixed(3), weights: w.slice() });
      chunkRewardSum = 0; chunkDeals = 0;
    }
  }

  // Early stopping: REINFORCE with annealed sigma is noisy late in training
  // (small sigma -> large 1/sigma^2 gradient scale -> occasional destabilizing
  // step even with clipping). Rather than fight that with more knobs, we
  // just checkpoint the weights from the best-performing chunk, which is
  // standard practice for policy-gradient training.
  const best = curve.reduce((a, b) => (b.avgDealScore > a.avgDealScore ? b : a));
  const weights = {};
  ['bias', ...FEATURE_NAMES].forEach((name, i) => { weights[name] = +best.weights[i].toFixed(4); });
  curve.forEach(c => delete c.weights);
  return { weights, curve, bestChunkThroughMatch: best.throughMatch };
}

if (require.main === module) {
  const numMatches = parseInt(process.argv[2], 10) || 4000;
  console.log(`Training RL bidder over ${numMatches} matches (${numMatches * 8 * 4} bid decisions)...\n`);
  const { weights, curve, bestChunkThroughMatch } = trainRL(numMatches);

  console.log('Learning curve (avg deal score per bid, per chunk):');
  curve.forEach(c => console.log(`  through match ${c.throughMatch}: ${c.avgDealScore}`));
  console.log(`\nCheckpointing weights from best chunk (through match ${bestChunkThroughMatch})`);

  console.log('\nFinal learned weights:');
  Object.entries(weights).forEach(([k, v]) => console.log(`  ${k.padEnd(16)} ${v}`));

  fs.writeFileSync('rl-bid-weights.json', JSON.stringify(weights, null, 2));
  console.log('\nWrote rl-bid-weights.json');
}

module.exports = { trainRL };
