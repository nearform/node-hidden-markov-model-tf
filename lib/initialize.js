
const ndarray = require('ndarray')
const kmeans = require('ml-kmeans');
const tf = require('@tensorflow/tfjs-core');

class Initialize {
  constructor({states, dimensions}) {
    this._states = states;
    this._dimensions = dimensions;
  }

  _reshapeTimeless2d(dataView) {
    // Convert Tensor into a 2D array (N * T, D), which should be used
    // by KMeans.
    const timeless2d = [];
    for (let t = 0; t < dataView.shape[0]; t++) {
      for (let n = 0; n < dataView.shape[1]; n++) {
        const observation = [];
        for (let d = 0; d < dataView.shape[2]; d++) {
          observation.push(dataView.get(t, n, d));
        }
        timeless2d.push(observation);
      }
    }

    return timeless2d;
  }

  _groupByIndex(timeless2d, index) {
    // Creates List of all groups
    const groups = [];
    for (let state = 0; state < this._states; state++) groups.push([]);

    // Assign data to the group specified by the index
    for (let i = 0; i < timeless2d.length; i++) {
      const centroidIndex = index[i];
      groups[centroidIndex].push(timeless2d[i]);
    }

    return groups;
  }

  async _computeCovariance(dataGrouped) {
    // Compute covariance for each group
    const SigmaRaw = await tf.tidy(() => {
      const SigmaList = [];
      for (let state = 0; state < this._states; state++) {
        const centroidData = tf.tensor2d(dataGrouped[state]);
        const mean = tf.mean(centroidData, 0, true);
        const x_m_mu = tf.sub(centroidData, mean);
        const unscaledCovariance = tf.matMul(x_m_mu, x_m_mu, true, false);
        const covariance = tf.div(
          unscaledCovariance,
          centroidData.shape[0] - 1
        );
        SigmaList.push(covariance);
      }
      return tf.stack(SigmaList);
    }).data();

    return SigmaRaw;
  }

  async compute(data, {maxIterations, tolerance, seed}) {
    const raw = await data.data();
    const dataView = ndarray(raw, data.shape);

    // Prepear data for kmeans
    const timeless2d = this._reshapeTimeless2d(dataView);

    // Use KMeans to intialize \mu and \Sigma
    const kmeansResult = kmeans(timeless2d, this._states, {
      maxIterations: maxIterations,
      tolerance: tolerance,
      seed: seed
    });

    // Compute \mu
    const kmeansCentroids = kmeansResult.centroids.map((o) => o.centroid);
    const mu = tf.tensor(kmeansCentroids);

    // Compute \Sigma
    const kmeansStates = kmeansResult.clusters;
    const dataGrouped = this._groupByIndex(timeless2d, kmeansStates);

    const SigmaRaw = await this._computeCovariance(dataGrouped);
    const Sigma = tf.tensor(
      SigmaRaw,
      [this._states, this._dimensions, this._dimensions],
      'float32'
    )

    return [mu, Sigma];
  }
}

module.exports = Initialize;
