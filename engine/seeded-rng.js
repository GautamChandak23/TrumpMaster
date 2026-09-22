/* ============================================================================
   seeded-rng.js
   A tiny, deterministic PRNG (mulberry32) used everywhere the engine needs
   randomness that must be reproducible: currently just the deck shuffle.

   This is the ONLY source of "game-critical" randomness in TrumpMaster.
   UI-only randomness (bot "thinking" delay, used purely for animation pacing)
   deliberately does NOT go through this — see rulesetVersion notes in
   RULES.md Appendix B.

   mulberry32(seed) returns a function with the same signature as
   Math.random(): call it with no args, get a float in [0, 1).
   Same seed -> same infinite sequence of outputs, on any machine, forever.
============================================================================ */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The ONE place true, non-reproducible entropy enters the system: picking a
// fresh seed for a brand-new match. Everything downstream of this call is
// deterministic given the seed.
function generateSeed() {
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    return crypto.getRandomValues(new Uint32Array(1))[0];
  }
  // Node fallback (or very old browsers)
  return Math.floor(Math.random() * 0xFFFFFFFF) >>> 0;
}

// Human-friendly, compact display form for a seed, e.g. "1D4F8AE2".
function seedToLabel(seed) {
  return (seed >>> 0).toString(16).toUpperCase().padStart(8, "0");
}

function labelToSeed(label) {
  return parseInt(label, 16) >>> 0;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { mulberry32, generateSeed, seedToLabel, labelToSeed };
}
