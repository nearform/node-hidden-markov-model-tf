
const ndarray = require('ndarray');
const kmeans = require('ml-kmeans');
const tf = require('@tensorflow/tfjs-core');

function diagonalTensor (value, dimensions) {
  const data = new Float32Array(dimensions * dimensions);
  for (let d = 0; d < dimensions; d++) {
    data[d * dimensions + d] = value;
  }
  return tf.tensor(data, [dimensions, dimensions]);
}

class Initialize {
  constructor ({ states, dimensions }) {
    this._states = states;
    this._dimensions = dimensions;
  }

  _reshapeTimeless2d (dataView) {
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

  _groupByIndex (timeless2d, index) {
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

  _computeCovariance (tensor) {
    const mean = tf.mean(tensor, 0, true);
    const x_m_mu = tf.sub(tensor, mean);
    const unscaled_covariance = tf.matMul(x_m_mu, x_m_mu, true, false);
    const covariance_unsafe = tf.div(
      unscaled_covariance,
      tensor.shape[0] - 1
    );
    const covariance_safe = tf.add(
      covariance_unsafe,
      diagonalTensor(1e-6, this._dimensions)
    );
    return covariance_safe;
  }

  async _computeStateCovariance (dataAll, dataGrouped) {
    // Compute covariance for each group
    const SigmaRaw = await tf.tidy(() => {
      const SigmaList = [];
      const dataAllTensor = tf.tensor2d(dataAll);
      for (let state = 0; state < this._states; state++) {
        if (dataGrouped[state].length <= 1) {
          SigmaList.push(this._computeCovariance(dataAllTensor));
        } else {
          SigmaList.push(this._computeCovariance(
            tf.tensor2d(dataGrouped[state])
          ));
        }
      }
      return tf.stack(SigmaList);
    }).data();

    return SigmaRaw;
  }

  async compute (data, { maxIterations, tolerance, seed }) {
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

    const SigmaRaw = await this._computeStateCovariance(
      timeless2d, dataGrouped
    );
    const Sigma = tf.tensor(
      SigmaRaw,
      [this._states, this._dimensions, this._dimensions],
      'float32'
    );

    return [mu, Sigma];
  }
}

module.exports = Initialize;
