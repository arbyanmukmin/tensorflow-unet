import * as tf from '@tensorflow/tfjs';
import { buildUNet } from './unet.js';

// [noisy = 3, albedo = 3, normal = 2, motion = 2, history = 3, history_mask = 1] total = 14
const modelBeautyTemporal = buildUNet([null, null, 14], 32, 4);
modelBeautyTemporal.summary();

// [noisy = 3, albedo = 3, normal = 2] total = 8
// const modelBeauty = buildUNet([null, null, 8], 32, 4);
// modelBeauty.summary();