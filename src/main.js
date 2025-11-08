import * as tf from '@tensorflow/tfjs';
import { buildUNet } from './unet.js';

const model = buildUNet([128, 128, 3], 32, 4);
model.summary();
