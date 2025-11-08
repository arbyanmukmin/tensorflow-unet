import * as tf from '@tensorflow/tfjs';

/**
 * Builds a simple scalable U-Net model.
 * @param {number[]} inputShape - [height, width, channels]
 * @param {number} baseFilters - number of filters in the first level
 * @param {number} depth - how many down/up levels
 * @returns {tf.LayersModel}
 */
export function buildUNet(inputShape = [128, 128, 3], baseFilters = 32, depth = 4) {
    const inputs = tf.input({ shape: inputShape });

    // Encoder
    let x = inputs;
    const skips = [];
    for(let i = 0; i < depth; i++) {
        const filters = baseFilters * Math.pow(2, i);
        x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
        x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
        skips.push(x);
        x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x);
    }

    // Bottleneck
    const bottleneckFilters = baseFilters * Math.pow(2, depth);
    x = tf.layers.conv2d({ filters: bottleneckFilters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
    x = tf.layers.conv2d({ filters: bottleneckFilters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);

    // Decoder
    for(let i = depth - 1; i >= 0; i--) {
        const filters = baseFilters * Math.pow(2, i);
        x = tf.layers.upSampling2d({ size: [2, 2] }).apply(x);
        x = tf.layers.concatenate().apply([x, skips[i]]);
        x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
        x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
    }

    const outputs = tf.layers.conv2d({ filters: 1, kernelSize: 1, activation: 'sigmoid' }).apply(x);
    return tf.model({ inputs, outputs });
}