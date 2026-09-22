/* ============================================================================
   fit-regression.js
   Fits an ordinary-least-squares linear regression (via the normal
   equations) of tricksWon on hand-shape features, using the CSV produced by
   collect-training-data.js. Prints fitted coefficients, R^2, and mean
   absolute error.

   Usage:
     node fit-regression.js [csvFile]
============================================================================ */
const fs = require('fs');

function parseCSV(path) {
  const lines = fs.readFileSync(path, 'utf8').trim().split('\n');
  const header = lines[0].split(',');
  const rows = lines.slice(1).map(l => l.split(',').map(Number));
  return { header, rows };
}

// Solve (X^T X) w = X^T y via Gaussian elimination.
function solve(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map(row => row[n]);
}

function fitOLS(rows, featureNames) {
  const nFeatures = featureNames.length + 1; // + bias
  const X = rows.map(r => [1, ...featureNames.map((_, i) => r[i])]);
  const y = rows.map(r => r[featureNames.length]);

  const XtX = Array.from({ length: nFeatures }, () => new Array(nFeatures).fill(0));
  const Xty = new Array(nFeatures).fill(0);
  X.forEach((xi, idx) => {
    for (let a = 0; a < nFeatures; a++) {
      Xty[a] += xi[a] * y[idx];
      for (let b = 0; b < nFeatures; b++) XtX[a][b] += xi[a] * xi[b];
    }
  });

  const w = solve(XtX, Xty);

  const yMean = y.reduce((s, v) => s + v, 0) / y.length;
  let ssRes = 0, ssTot = 0, sumAbsErr = 0;
  X.forEach((xi, idx) => {
    const pred = xi.reduce((s, v, i) => s + v * w[i], 0);
    ssRes += (y[idx] - pred) ** 2;
    ssTot += (y[idx] - yMean) ** 2;
    sumAbsErr += Math.abs(y[idx] - pred);
  });
  const r2 = 1 - ssRes / ssTot;
  const mae = sumAbsErr / y.length;

  return { weights: w, r2, mae, n: rows.length };
}

if (require.main === module) {
  const csvFile = process.argv[2] || 'training-data.csv';
  const { header, rows } = parseCSV(csvFile);
  const featureNames = header.slice(0, -1); // all but tricksWon

  const { weights, r2, mae, n } = fitOLS(rows, featureNames);

  console.log(`Fitted on ${n} samples from ${csvFile}\n`);
  console.log('bias'.padEnd(16), weights[0].toFixed(4));
  featureNames.forEach((name, i) => console.log(name.padEnd(16), weights[i + 1].toFixed(4)));
  console.log(`\nR^2 = ${r2.toFixed(3)}   Mean |pred - actual| = ${mae.toFixed(3)} tricks`);
}

module.exports = { fitOLS, parseCSV };
