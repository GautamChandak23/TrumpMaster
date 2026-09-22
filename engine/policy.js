/**
 * policy.js
 * =========
 *
 * Pure-JS re-implementation of policy_network.py's forward pass,
 * for running the trained TrumpMaster policy client-side in the
 * browser with no server and no PyTorch.
 *
 * Architecture (must match the exported checkpoint exactly):
 *
 *   obs (207,) -> Linear(207,256) -> ReLU -> Linear(256,256) -> ReLU
 *       -> policy_head: Linear(256,52)   (masked categorical logits)
 *       -> value_head:  Linear(256,1)
 *
 * Usage
 * -----
 *   const policy = await TrumpMasterPolicy.load('phase2_policy_best.json');
 *   const { action, probs, value } = policy.act(obsArray, legalMaskArray);
 *   // action: integer in [0,52) — pass through the SAME index->card
 *   // decoding your JS game engine's action space uses (must match
 *   // action_mask.py's ActionMask.decode_action encoding).
 */

class TrumpMasterPolicy {

  constructor(weights) {
    this.obsDim = weights.obs_dim;
    this.hiddenDim = weights.hidden_dim;
    this.actionDim = weights.action_dim;

    this.trunk0W = weights.trunk0_weight;   // [hidden][obs]
    this.trunk0B = weights.trunk0_bias;     // [hidden]
    this.trunk2W = weights.trunk2_weight;   // [hidden][hidden]
    this.trunk2B = weights.trunk2_bias;     // [hidden]
    this.policyW = weights.policy_head_weight; // [action][hidden]
    this.policyB = weights.policy_head_bias;   // [action]
    this.valueW = weights.value_head_weight;   // [1][hidden]
    this.valueB = weights.value_head_bias;     // [1]
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to load policy weights from ${url}: ${res.status}`);
    }
    const weights = await res.json();
    return new TrumpMasterPolicy(weights);
  }

  static fromObject(weights) {
    return new TrumpMasterPolicy(weights);
  }

  // y = W @ x + b, with W as [out][in] (nn.Linear layout), followed
  // by an optional ReLU.
  _linear(x, W, b, relu) {
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      const row = W[i];
      let sum = b[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * x[j];
      }
      out[i] = relu ? Math.max(0, sum) : sum;
    }
    return out;
  }

  /**
   * Runs the trunk + both heads.
   * obs: number[207]
   * Returns { logits: number[52], value: number }
   */
  forward(obs) {
    if (obs.length !== this.obsDim) {
      throw new Error(
        `Observation length ${obs.length} != expected ${this.obsDim}`
      );
    }

    const h1 = this._linear(obs, this.trunk0W, this.trunk0B, true);
    const h2 = this._linear(h1, this.trunk2W, this.trunk2B, true);

    const logits = this._linear(h2, this.policyW, this.policyB, false);
    const valueArr = this._linear(h2, this.valueW, this.valueB, false);

    return { logits, value: valueArr[0] };
  }

  /**
   * Applies the legal-action mask the same way policy_network.py's
   * masked_logits does: illegal actions get -Infinity so softmax
   * assigns them ~0 probability.
   *
   * mask: number[52] or boolean[52], 1/true = legal, 0/false = illegal.
   */
  _maskedSoftmax(logits, mask) {
    const masked = logits.map((v, i) => (mask[i] ? v : -Infinity));

    const maxLogit = Math.max(...masked);
    const exps = masked.map((v) =>
      v === -Infinity ? 0 : Math.exp(v - maxLogit)
    );
    const sum = exps.reduce((a, b) => a + b, 0);

    return exps.map((v) => v / sum);
  }

  /**
   * Chooses an action given an observation and a legal-action mask.
   *
   * deterministic: true = argmax (recommended for a deployed
   * opponent — matches how the Python eval routines picked moves).
   * false = samples from the masked softmax distribution.
   *
   * Returns { action, probs, value }.
   */
  act(obs, mask, deterministic = true) {
    const { logits, value } = this.forward(obs);
    const probs = this._maskedSoftmax(logits, mask);

    let action;
    if (deterministic) {
      action = probs.indexOf(Math.max(...probs));
    } else {
      const r = Math.random();
      let cumulative = 0;
      action = probs.length - 1;
      for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (r < cumulative) {
          action = i;
          break;
        }
      }
    }

    return { action, probs, value };
  }
}

// Support both browser <script> global usage and ES module import.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { TrumpMasterPolicy };
}
