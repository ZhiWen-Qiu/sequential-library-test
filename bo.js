// Sequential plane search (Koyama, Sato, Goto 2020) — plain JavaScript, no dependencies.
// Design space is the unit box [0,1]^n. Preferences use the Bradley-Terry-Luce model,
// the goodness function has a Gaussian-process prior (ARD Matern 5/2), and the
// posterior is approximated with the Laplace method.
// Simplification vs. the paper: kernel hyperparameters are fixed instead of MAP-estimated.

// Wrapped in a function so its names never clash with the page that loads it.
(function () {
"use strict";

const AMPLITUDE = 0.5;     // kernel variance
const LENGTH_SCALE = 0.5;  // same for every dimension
const BTL_SCALE = 0.01;
const JITTER = 1e-6;

// ---------- small linear algebra helpers ----------

function zeros(r, c) {
  const m = new Array(r);
  for (let i = 0; i < r; i++) m[i] = new Float64Array(c);
  return m;
}

function identity(n) {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m[i][i] = 1;
  return m;
}

function matMul(A, B) {
  const r = A.length, k = B.length, c = B[0].length;
  const C = zeros(r, c);
  for (let i = 0; i < r; i++) {
    const Ai = A[i], Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < c; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}

function matVec(A, x) {
  const y = new Float64Array(A.length);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    const Ai = A[i];
    for (let j = 0; j < x.length; j++) s += Ai[j] * x[j];
    y[i] = s;
  }
  return y;
}

// Solves A X = B (B is a matrix) by LU with partial pivoting. A and B are not modified.
function solve(A, B) {
  const n = A.length, c = B[0].length;
  const M = A.map(row => Float64Array.from(row));
  const X = B.map(row => Float64Array.from(row));
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    if (p !== k) {
      [M[k], M[p]] = [M[p], M[k]];
      [X[k], X[p]] = [X[p], X[k]];
    }
    const piv = M[k][k] || 1e-12;
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / piv;
      if (f === 0) continue;
      for (let j = k; j < n; j++) M[i][j] -= f * M[k][j];
      for (let j = 0; j < c; j++) X[i][j] -= f * X[k][j];
    }
  }
  for (let k = n - 1; k >= 0; k--) {
    const piv = M[k][k] || 1e-12;
    for (let j = 0; j < c; j++) {
      let s = X[k][j];
      for (let i = k + 1; i < n; i++) s -= M[k][i] * X[i][j];
      X[k][j] = s / piv;
    }
  }
  return X;
}

function solveVec(A, b) {
  return Float64Array.from(solve(A, Array.from(b, v => Float64Array.of(v))), row => row[0]);
}

// ---------- vectors ----------

const clamp01 = v => Math.min(1, Math.max(0, v));
const clampVec = x => Float64Array.from(x, clamp01);
const add = (a, b, s = 1) => Float64Array.from(a, (v, i) => v + s * b[i]);
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = a => Math.sqrt(dot(a, a));
const scale = (a, s) => Float64Array.from(a, v => v * s);

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const randnVec = n => Float64Array.from({ length: n }, randn);

// Largest t >= 0 such that c + t*d and c - t*d both stay inside [0,1]^n.
function maxSymmetricStep(c, d) {
  let t = Infinity;
  for (let i = 0; i < c.length; i++) {
    if (Math.abs(d[i]) < 1e-12) continue;
    t = Math.min(t, Math.min(c[i], 1 - c[i]) / Math.abs(d[i]));
  }
  return t;
}

// ---------- Gaussian process with BTL preferences ----------

function kernel(a, b) {
  let r2 = 0;
  for (let i = 0; i < a.length; i++) {
    const d = (a[i] - b[i]) / LENGTH_SCALE;
    r2 += d * d;
  }
  const r = Math.sqrt(r2), s5r = Math.sqrt(5) * r;
  return AMPLITUDE * (1 + s5r + (5 / 3) * r2) * Math.exp(-s5r);
}

function normalPdf(z) { return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); }

function normalCdf(z) {
  // Abramowitz-Stegun 7.1.26 approximation of erf
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

class PreferenceModel {
  constructor(dim) {
    this.dim = dim;
    this.points = [];   // Float64Array[]
    this.prefs = [];    // { chosen: index, others: index[] }
    this.fitted = null;
  }

  indexOf(x) {
    for (let i = 0; i < this.points.length; i++) {
      let d = 0;
      for (let k = 0; k < this.dim; k++) d = Math.max(d, Math.abs(this.points[i][k] - x[k]));
      if (d < 1e-6) return i;
    }
    this.points.push(Float64Array.from(x));
    return this.points.length - 1;
  }

  // "chosen is better than every option in others"
  addPreference(chosen, others) {
    const c = this.indexOf(chosen);
    const o = [...new Set(others.map(x => this.indexOf(x)))].filter(i => i !== c);
    if (o.length > 0) this.prefs.push({ chosen: c, others: o });
    this.fitted = null;
  }

  fit() {
    const m = this.points.length;
    const K = zeros(m, m);
    for (let i = 0; i < m; i++) {
      for (let j = i; j < m; j++) K[i][j] = K[j][i] = kernel(this.points[i], this.points[j]);
      K[i][i] += JITTER;
    }

    // Newton iterations for the posterior mode: f <- K (I + W K)^-1 (W f + g)
    let f = new Float64Array(m);
    let W = zeros(m, m), g = new Float64Array(m);
    for (let iter = 0; iter < 50; iter++) {
      ({ W, g } = this.likelihoodDerivatives(f));
      const Wf = matVec(W, f);
      const rhs = Float64Array.from(Wf, (v, i) => v + g[i]);
      const B = matMul(W, K);
      for (let i = 0; i < m; i++) B[i][i] += 1;
      const fNew = matVec(K, solveVec(B, rhs));
      let change = 0;
      for (let i = 0; i < m; i++) change = Math.max(change, Math.abs(fNew[i] - f[i]));
      f = fNew;
      if (change < 1e-6) break;
    }
    ({ W } = this.likelihoodDerivatives(f));

    const alpha = solveVec(K, f);
    // Predictive variance uses (K + W^-1)^-1 = (I + W K)^-1 W, which avoids inverting W.
    const B = matMul(W, K);
    for (let i = 0; i < m; i++) B[i][i] += 1;
    const Mvar = solve(B, W);

    this.fitted = { alpha, Mvar, f };
  }

  likelihoodDerivatives(f) {
    const m = f.length;
    const W = zeros(m, m), g = new Float64Array(m);
    for (const { chosen, others } of this.prefs) {
      const idx = [chosen, ...others];
      const maxF = Math.max(...idx.map(i => f[i] / BTL_SCALE));
      const e = idx.map(i => Math.exp(f[i] / BTL_SCALE - maxF));
      const sum = e.reduce((a, b) => a + b, 0);
      const p = e.map(v => v / sum);
      idx.forEach((i, a) => {
        g[i] += ((i === chosen ? 1 : 0) - p[a]) / BTL_SCALE;
        idx.forEach((j, b) => {
          W[i][j] += ((a === b ? p[a] : 0) - p[a] * p[b]) / (BTL_SCALE * BTL_SCALE);
        });
      });
    }
    return { W, g };
  }

  predict(x) {
    if (!this.fitted) this.fit();
    const { alpha, Mvar } = this.fitted;
    const k = Float64Array.from(this.points, p => kernel(p, x));
    const mean = dot(k, alpha);
    const variance = Math.max(1e-12, AMPLITUDE - dot(k, matVec(Mvar, k)));
    return { mean, sd: Math.sqrt(variance) };
  }

  expectedImprovement(x, bestMean) {
    const { mean, sd } = this.predict(x);
    const z = (mean - bestMean) / sd;
    return sd * (z * normalCdf(z) + normalPdf(z));
  }
}

// ---------- plane construction ----------

// Grid coordinates (s, t) in [-1,1]^2 map onto the rhombus with vertices c±u, c±v.
function planePoint(plane, s, t) {
  const a = (s + t) / 2, b = (s - t) / 2;
  const x = new Float64Array(plane.c.length);
  for (let i = 0; i < x.length; i++) x[i] = clamp01(plane.c[i] + a * plane.u[i] + b * plane.v[i]);
  return x;
}

function randomOrthogonal(u) {
  const n = u.length;
  let v = randnVec(n);
  const uu = dot(u, u);
  if (uu > 1e-12) v = add(v, u, -dot(v, u) / uu);
  return scale(v, 1 / (norm(v) || 1));
}

function initialPlane(dim, center, size = 0.5) {
  const c = center ? clampVec(center) : new Float64Array(dim).fill(0.5);
  const u = scale(randomOrthogonal(new Float64Array(dim)), size);
  const v = scale(randomOrthogonal(u), size);
  return { c, u, v };
}

class SequentialPlaneSearch {
  // center: optional starting point in [0,1]^dim (defaults to the middle of the space)
  // size: how far the first set of options spreads from the center
  constructor(dim, center, size) {
    this.dim = dim;
    this.model = new PreferenceModel(dim);
    this.plane = initialPlane(dim, center, size);
  }

  // A fresh set of random directions around `center`, keeping what was learned so far.
  // A bigger `size` reaches further away from the center.
  reseed(center, size) {
    this.plane = initialPlane(this.dim, center, size);
    return this.plane;
  }

  // "None of these": the current choice is better than everything shown on this plane.
  // The next plane keeps the same reach but points in new directions, away from the ones just shown.
  // `shown`: the options that were on screen (defaults to the plane's corners).
  dislike(current, size, shown) {
    const { c, u, v } = this.plane;
    const rejected = (shown && shown.length ? shown : [add(c, u), add(c, u, -1), add(c, v), add(c, v, -1)]).map(clampVec);
    this.model.addPreference(clampVec(current), rejected);
    const away = [u, v];
    const newDir = () => {
      let d = randnVec(this.dim);
      for (const a of away) { const aa = dot(a, a); if (aa > 1e-12) d = add(d, a, -dot(d, a) / aa); }
      return scale(d, 1 / (norm(d) || 1));
    };
    const nu = newDir();
    away.push(nu);
    const nv = newDir();
    this.plane = { c: clampVec(current), u: scale(nu, size), v: scale(nv, size) };
    return this.plane;
  }

  // Everything needed to go back to this exact state later (for "undo").
  snapshot() {
    const copyPlane = p => ({ c: Float64Array.from(p.c), u: Float64Array.from(p.u), v: Float64Array.from(p.v) });
    return {
      points: this.model.points.map(p => Float64Array.from(p)),
      prefs: this.model.prefs.map(p => ({ chosen: p.chosen, others: [...p.others] })),
      plane: copyPlane(this.plane),
      copyPlane,
    };
  }

  restore(snap) {
    this.model.points = snap.points.map(p => Float64Array.from(p));
    this.model.prefs = snap.prefs.map(p => ({ chosen: p.chosen, others: [...p.others] }));
    this.model.fitted = null;
    this.plane = snap.copyPlane(snap.plane);
  }

  // The parameter set the user picked on the current plane.
  // Optional extra feedback makes the model learn faster:
  //   opts.liked / opts.disliked: options the user marked; opts.neutral: unmarked options that were shown
  //   (all better than disliked, worse than liked). opts.reach: fixed size for the next plane.
  submit(chosen, opts = {}) {
    const { c, u, v } = this.plane;
    const liked = (opts.liked || []).map(clampVec);
    const disliked = (opts.disliked || []).map(clampVec);
    const neutral = (opts.neutral && opts.neutral.length ? opts.neutral : [c, add(c, u), add(c, u, -1), add(c, v), add(c, v, -1)]).map(clampVec);
    const best = clampVec(chosen);
    // 最喜歡 > 喜歡 > 沒意見 > 不喜歡
    this.model.addPreference(best, [...liked, ...neutral, ...disliked]);
    if (neutral.length || disliked.length) for (const l of liked) this.model.addPreference(l, [...neutral, ...disliked]);
    if (disliked.length) for (const n of neutral) this.model.addPreference(n, disliked);
    this.plane = this.nextPlane(best, opts.reach);
    return this.plane;
  }

  nextPlane(xBest, reach) {
    const model = this.model;
    model.fit();
    const bestMean = model.predict(xBest).mean;
    const ei = x => model.expectedImprovement(x, bestMean);

    const xEI = this.maximizeEI(ei);
    let u = add(xEI, xBest, -1);
    if (norm(u) < 0.05) u = scale(randomOrthogonal(new Float64Array(this.dim)), 0.15);
    if (reach) u = scale(u, reach / norm(u)); // keep BO's direction, use the requested size

    const lattice = [];
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) lattice.push([-1 + i / 2, -1 + j / 2]);
    const planeScore = v => {
      const plane = { c: xBest, u, v };
      let s = 0;
      for (const [a, b] of lattice) s += ei(planePoint(plane, a, b));
      return s / lattice.length;
    };

    const uLen = norm(u);
    const candidateV = () => {
      const dir = randomOrthogonal(u);
      if (reach) return scale(dir, reach);
      const maxLen = Math.min(2 * uLen, maxSymmetricStep(xBest, dir));
      const minLen = Math.min(Math.max(0.5 * uLen, 0.1), maxLen);
      return scale(dir, minLen + Math.random() * (maxLen - minLen));
    };

    let bestV = candidateV(), bestScore = planeScore(bestV);
    for (let i = 0; i < 150; i++) {
      const v = candidateV(), s = planeScore(v);
      if (s > bestScore) { bestV = v; bestScore = s; }
    }
    // local refinement that keeps v orthogonal to u
    let step = 0.3;
    for (let i = 0; i < 80; i++) {
      let v = add(bestV, scale(randnVec(this.dim), step * norm(bestV)));
      v = add(v, u, -dot(v, u) / dot(u, u));
      if (reach) v = scale(v, reach / (norm(v) || 1));
      const s = planeScore(v);
      if (s > bestScore) { bestV = v; bestScore = s; } else step *= 0.97;
    }
    return { c: xBest, u, v: bestV };
  }

  maximizeEI(ei) {
    const dim = this.dim;
    const candidates = [];
    for (let i = 0; i < 1500; i++) candidates.push(Float64Array.from({ length: dim }, Math.random));
    for (const p of this.model.points.slice(-30)) {
      for (let i = 0; i < 30; i++) candidates.push(clampVec(add(p, randnVec(dim), 0.1)));
    }
    const scored = candidates.map(x => ({ x, s: ei(x) })).sort((a, b) => b.s - a.s).slice(0, 5);

    let best = scored[0];
    for (const start of scored) {
      let cur = start, step = 0.1;
      for (let i = 0; i < 60; i++) {
        const x = clampVec(add(cur.x, randnVec(dim), step));
        const s = ei(x);
        if (s > cur.s) cur = { x, s }; else step *= 0.95;
      }
      if (cur.s > best.s) best = cur;
    }
    return best.x;
  }
}

// CommonJS for the Photoshop plugin, a global for the web page.
if (typeof module !== "undefined" && module.exports) module.exports = { SequentialPlaneSearch, planePoint };
else globalThis.SequentialGallery = { SequentialPlaneSearch, planePoint };
})();
