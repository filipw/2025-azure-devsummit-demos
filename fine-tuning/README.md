# Phi fine tuning with MLX

## Prerequisites 

Install MLX (included in the `requirements.txt` file of this repo).

```
pip install -r requirements.txt
```

## 1. Sample task

Fine tuning Phi to become a music library controller, capable of invoking the following functions

* play_song(title)
* play_playlist(title)
* pause
* stop
* next track
* previous track
* volume up
* volume down
* mute
* unmute

## 2. Prepare test data

Sample `./data/train.jsonl`:

```json
{"text": "<|user|>Play Bohemian Rhapsody<|end|>\n<|assistant|>fn:play_song \"bohemian rhapsody\"<|end|>"}
{"text": "<|user|>Start my workout playlist<|end|>\n<|assistant|>fn:play_list \"workout mix\"<|end|>"}
{"text": "<|user|>Next song<|end|>\n<|assistant|>fn:next<|end|>"}
{"text": "<|user|>Skip track<|end|>\n<|assistant|>fn:next<|end|>"}
{"text": "<|user|>Make it louder<|end|>\n<|assistant|>fn:vol_up<|end|>"}
{"text": "<|user|>Turn down volume<|end|>\n<|assistant|>fn:vol_down<|end|>"}
```

## 3. Run LoRa fine tuning

```
mlx_lm.lora --model microsoft/Phi-3.5-mini-instruct --train --data ./data --iters 500
```

With the included data set this runs at most few minutes. It would be longer if each line is extended for more tokens.

Expected output should be similar too:

```
Loading datasets
Training
Trainable parameters: 0.041% (1.573M/3821.080M)
Starting training..., iters: 500
Calculating loss...: 100%|██████████████████████████████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.65it/s]
Iter 1: Val loss 14.555, Val took 7.179s
Iter 10: Train loss 10.986, Learning Rate 1.000e-05, It/sec 1.961, Tokens/sec 129.632, Trained Tokens 661, Peak mem 8.115 GB
Iter 20: Train loss 5.275, Learning Rate 1.000e-05, It/sec 2.323, Tokens/sec 153.066, Trained Tokens 1320, Peak mem 8.115 GB
Iter 30: Train loss 4.134, Learning Rate 1.000e-05, It/sec 2.338, Tokens/sec 134.440, Trained Tokens 1895, Peak mem 8.115 GB
Iter 40: Train loss 2.653, Learning Rate 1.000e-05, It/sec 2.338, Tokens/sec 138.864, Trained Tokens 2489, Peak mem 8.115 GB
Iter 50: Train loss 1.915, Learning Rate 1.000e-05, It/sec 2.230, Tokens/sec 124.001, Trained Tokens 3045, Peak mem 8.115 GB
Iter 60: Train loss 1.628, Learning Rate 1.000e-05, It/sec 2.224, Tokens/sec 147.891, Trained Tokens 3710, Peak mem 8.115 GB
Iter 70: Train loss 1.600, Learning Rate 1.000e-05, It/sec 2.201, Tokens/sec 143.723, Trained Tokens 4363, Peak mem 8.115 GB
Iter 80: Train loss 1.356, Learning Rate 1.000e-05, It/sec 2.302, Tokens/sec 146.176, Trained Tokens 4998, Peak mem 8.115 GB
Iter 90: Train loss 1.259, Learning Rate 1.000e-05, It/sec 2.239, Tokens/sec 132.969, Trained Tokens 5592, Peak mem 8.115 GB
Iter 100: Train loss 1.267, Learning Rate 1.000e-05, It/sec 2.284, Tokens/sec 130.648, Trained Tokens 6164, Peak mem 8.115 GB
Iter 100: Saved adapter weights to adapters/adapters.safetensors and adapters/0000100_adapters.safetensors.
Iter 110: Train loss 1.184, Learning Rate 1.000e-05, It/sec 2.277, Tokens/sec 139.563, Trained Tokens 6777, Peak mem 8.122 GB
Iter 120: Train loss 1.174, Learning Rate 1.000e-05, It/sec 2.277, Tokens/sec 136.386, Trained Tokens 7376, Peak mem 8.122 GB
Iter 130: Train loss 1.242, Learning Rate 1.000e-05, It/sec 2.263, Tokens/sec 155.495, Trained Tokens 8063, Peak mem 8.122 GB
Iter 140: Train loss 1.210, Learning Rate 1.000e-05, It/sec 2.217, Tokens/sec 133.239, Trained Tokens 8664, Peak mem 8.122 GB
Iter 150: Train loss 1.200, Learning Rate 1.000e-05, It/sec 2.267, Tokens/sec 144.153, Trained Tokens 9300, Peak mem 8.122 GB
Iter 160: Train loss 1.063, Learning Rate 1.000e-05, It/sec 2.310, Tokens/sec 139.531, Trained Tokens 9904, Peak mem 8.122 GB
Iter 170: Train loss 1.007, Learning Rate 1.000e-05, It/sec 2.328, Tokens/sec 160.369, Trained Tokens 10593, Peak mem 8.122 GB
Iter 180: Train loss 1.020, Learning Rate 1.000e-05, It/sec 2.279, Tokens/sec 143.580, Trained Tokens 11223, Peak mem 8.122 GB
Iter 190: Train loss 1.089, Learning Rate 1.000e-05, It/sec 2.278, Tokens/sec 142.845, Trained Tokens 11850, Peak mem 8.122 GB
Calculating loss...: 100%|██████████████████████████████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.53it/s]
Iter 200: Val loss 1.045, Val took 5.383s
Iter 200: Train loss 1.027, Learning Rate 1.000e-05, It/sec 2.301, Tokens/sec 147.020, Trained Tokens 12489, Peak mem 8.122 GB
Iter 200: Saved adapter weights to adapters/adapters.safetensors and adapters/0000200_adapters.safetensors.
Iter 210: Train loss 1.190, Learning Rate 1.000e-05, It/sec 2.320, Tokens/sec 121.594, Trained Tokens 13013, Peak mem 8.122 GB
Iter 220: Train loss 0.994, Learning Rate 1.000e-05, It/sec 2.329, Tokens/sec 145.073, Trained Tokens 13636, Peak mem 8.122 GB
Iter 230: Train loss 0.992, Learning Rate 1.000e-05, It/sec 2.243, Tokens/sec 129.634, Trained Tokens 14214, Peak mem 8.122 GB
Iter 240: Train loss 0.982, Learning Rate 1.000e-05, It/sec 2.279, Tokens/sec 153.602, Trained Tokens 14888, Peak mem 8.122 GB
Iter 250: Train loss 0.934, Learning Rate 1.000e-05, It/sec 2.208, Tokens/sec 133.362, Trained Tokens 15492, Peak mem 8.122 GB
Iter 260: Train loss 1.039, Learning Rate 1.000e-05, It/sec 2.180, Tokens/sec 140.603, Trained Tokens 16137, Peak mem 8.122 GB
Iter 270: Train loss 1.024, Learning Rate 1.000e-05, It/sec 2.159, Tokens/sec 137.309, Trained Tokens 16773, Peak mem 8.122 GB
Iter 280: Train loss 1.004, Learning Rate 1.000e-05, It/sec 2.115, Tokens/sec 122.643, Trained Tokens 17353, Peak mem 8.122 GB
Iter 290: Train loss 1.025, Learning Rate 1.000e-05, It/sec 2.266, Tokens/sec 148.442, Trained Tokens 18008, Peak mem 8.122 GB
Iter 300: Train loss 0.954, Learning Rate 1.000e-05, It/sec 2.265, Tokens/sec 151.310, Trained Tokens 18676, Peak mem 8.122 GB
Iter 300: Saved adapter weights to adapters/adapters.safetensors and adapters/0000300_adapters.safetensors.
Iter 310: Train loss 0.922, Learning Rate 1.000e-05, It/sec 2.285, Tokens/sec 128.172, Trained Tokens 19237, Peak mem 8.122 GB
Iter 320: Train loss 0.858, Learning Rate 1.000e-05, It/sec 2.324, Tokens/sec 146.870, Trained Tokens 19869, Peak mem 8.122 GB
Iter 330: Train loss 0.914, Learning Rate 1.000e-05, It/sec 2.322, Tokens/sec 150.015, Trained Tokens 20515, Peak mem 8.122 GB
Iter 340: Train loss 1.000, Learning Rate 1.000e-05, It/sec 2.327, Tokens/sec 150.784, Trained Tokens 21163, Peak mem 8.122 GB
Iter 350: Train loss 0.985, Learning Rate 1.000e-05, It/sec 2.323, Tokens/sec 126.389, Trained Tokens 21707, Peak mem 8.122 GB
Iter 360: Train loss 0.913, Learning Rate 1.000e-05, It/sec 2.329, Tokens/sec 130.421, Trained Tokens 22267, Peak mem 8.122 GB
Iter 370: Train loss 0.853, Learning Rate 1.000e-05, It/sec 2.314, Tokens/sec 153.155, Trained Tokens 22929, Peak mem 8.122 GB
Iter 380: Train loss 0.891, Learning Rate 1.000e-05, It/sec 2.325, Tokens/sec 152.307, Trained Tokens 23584, Peak mem 8.122 GB
Iter 390: Train loss 0.857, Learning Rate 1.000e-05, It/sec 2.163, Tokens/sec 131.937, Trained Tokens 24194, Peak mem 8.122 GB
Calculating loss...: 100%|██████████████████████████████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.44it/s]
Iter 400: Val loss 0.983, Val took 5.529s
Iter 400: Train loss 1.008, Learning Rate 1.000e-05, It/sec 2.215, Tokens/sec 151.975, Trained Tokens 24880, Peak mem 8.122 GB
Iter 400: Saved adapter weights to adapters/adapters.safetensors and adapters/0000400_adapters.safetensors.
Iter 410: Train loss 0.884, Learning Rate 1.000e-05, It/sec 2.261, Tokens/sec 141.102, Trained Tokens 25504, Peak mem 8.122 GB
Iter 420: Train loss 0.876, Learning Rate 1.000e-05, It/sec 2.318, Tokens/sec 141.196, Trained Tokens 26113, Peak mem 8.122 GB
Iter 430: Train loss 0.860, Learning Rate 1.000e-05, It/sec 2.255, Tokens/sec 134.178, Trained Tokens 26708, Peak mem 8.122 GB
Iter 440: Train loss 0.905, Learning Rate 1.000e-05, It/sec 2.270, Tokens/sec 151.208, Trained Tokens 27374, Peak mem 8.122 GB
Iter 450: Train loss 0.950, Learning Rate 1.000e-05, It/sec 2.293, Tokens/sec 126.339, Trained Tokens 27925, Peak mem 8.122 GB
Iter 460: Train loss 0.946, Learning Rate 1.000e-05, It/sec 2.320, Tokens/sec 138.481, Trained Tokens 28522, Peak mem 8.122 GB
Iter 470: Train loss 0.907, Learning Rate 1.000e-05, It/sec 2.309, Tokens/sec 133.453, Trained Tokens 29100, Peak mem 8.122 GB
Iter 480: Train loss 0.868, Learning Rate 1.000e-05, It/sec 2.311, Tokens/sec 138.902, Trained Tokens 29701, Peak mem 8.122 GB
Iter 490: Train loss 0.771, Learning Rate 1.000e-05, It/sec 2.224, Tokens/sec 156.346, Trained Tokens 30404, Peak mem 8.122 GB
Calculating loss...: 100%|██████████████████████████████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.51it/s]
Iter 500: Val loss 0.958, Val took 5.410s
Iter 500: Train loss 0.757, Learning Rate 1.000e-05, It/sec 2.228, Tokens/sec 159.774, Trained Tokens 31121, Peak mem 8.122 GB
Iter 500: Saved adapter weights to adapters/adapters.safetensors and adapters/0000500_adapters.safetensors.
Saved final weights to adapters/adapters.safetensors.
```

### Loss progression

 - Starting val loss: 14.555 (but drops quickly)
 - Final val loss: 0.958 (very good)
 - Training loss: Stabilizes around 0.7–1.0 in later iterations
 - Val loss: Consistent and tracks training loss well, no signs of overfitting

### Key observations:

 - Rapid initial improvement: Loss drops from 14.6 to ~1.0 in first 100 iterations
 - Stabilization: By iteration 200, losses are low and stable
 - Generalization: Validation loss closely follows training loss, indicating good generalization
 - Final training loss around 0.75 is good for this type of task

### Summary

Given these results:

* The model has trained successfully
* 500 iterations was a good choice - we could have even have stopped at 300-400 iterations
* The stable validation loss suggests the model should generalize well

The output is created into `./adapters` folder and is an adapter which can be layered on top of the model.

It should be about 30 MB in size.

## 4. Test the adapter

The base model can be invoked with the adapter to test the fine tuning.

```
mlx_lm.generate --model microsoft/Phi-3.5-mini-instruct --adapter-path ./adapters --max-token 2048 --prompt "i don't like this song" --extra-eos-token "<|end|>" --temp 0.0
```

Expected output:

```
==========
fn:next
==========
Prompt: 10 tokens, 9.669 tokens-per-sec
Generation: 4 tokens, 25.524 tokens-per-sec
Peak memory: 7.868 GB
```

## 5. Merge the adapter into the model

```
mlx_lm.fuse --model microsoft/Phi-3.5-mini-instruct
```

This creates a fused safe tensors model inside `./fused_model` folder. From there it can be used directly with any ML framework that supports safe tensors, or it can be subject to quantization or other optimizations. 

## 6. Test the final model

There is a simple validation script in the repo which runs the inference using data from `./data/valid.jsonl`. This is a data set that the model has not seen before.

```
python validate.py
```

The script execute the inference against the fine tuned model (without the system instruction) and the base model with a system instruction and a few shot learning approach. The result should show that the fine tuned model is not only faster than the base model (as it uses less tokens!) but also dramatically more accurate than the base model.

The output should be similar to:

```
=== Starting Validation ===
Loading 25 validation examples...

=== Loading Fine-tuned Model ===
Fetching 13 files: 100%|██████████████████████████████████████████████████| 13/13 [00:00<00:00, 74387.38it/s]

=== Testing Fine-tuned Model ===
[ 1/25] ✅ 'Boost the volume' → fn:vol_up
[ 2/25] ✅ 'What's your name again?' → Sorry I cannot help with that
[ 3/25] ✅ 'Turn up' → fn:vol_up
[ 4/25] ✅ 'Lower it' → fn:vol_down
[ 5/25] ✅ 'Last song please' → fn:prev
[ 6/25] ✅ 'Skip' → fn:next
[ 7/25] ✅ 'Stop all' → fn:stop
[ 8/25] ✅ 'Audio enable' → fn:unmute
[ 9/25] ✅ 'Play We Will Rock You' → fn:play_song "we will rock you"
[10/25] ✅ 'Play Wish You Were Here' → fn:play_song "wish you were here"
[11/25] ✅ 'Forward' → fn:next
[12/25] ✅ 'Play Faithfully' → fn:play_song "faithfully"
[13/25] ✅ 'Play Open Arms' → fn:play_song "open arms"
[14/25] ✅ 'Play my lounge playlist' → fn:play_list "lounge"
[15/25] ✅ 'Switch' → fn:next
[16/25] ✅ 'Back' → fn:prev
[17/25] ✅ 'Reverse' → fn:prev
[18/25] ✅ 'End' → fn:stop
[19/25] ✅ 'Finish it' → fn:stop
[20/25] ✅ 'Pause now' → fn:pause
[21/25] ✅ 'Enough music' → fn:stop
[22/25] ✅ 'Silent' → fn:mute
[23/25] ✅ 'No noise' → fn:mute
[24/25] ✅ 'What's your age?' → Sorry I cannot help with that
[25/25] ✅ 'Enable audio' → fn:unmute

Summary: 25/25 correct (100.0%), avg 1.3s per request

==================================================

=== Testing Base Model ===
[ 1/25] ✅ 'Boost the volume' → fn:vol_up
[ 2/25] ❌ 'What's your name again?' → fn:no_function "name" (since I'm Phi, an AI language
[ 3/25] ❌ 'Turn up' → Sorry, I cannot execute actions. However, if you were using a text-based interface, you would typically input the command as described. For example: "Increase volume." In a real scenario,
[ 4/25] ❌ 'Lower it' → Sorry, I cannot adjust the volume directly. Please use the volume control on your device. If you have a remote or control panel, look for the volume down button or function to adjust the volume.
[ 5/25] ❌ 'Last song please' → fn:play_song "Clair de Lune" Deb
[ 6/25] ❌ 'Skip' → fn:vol_up
[ 7/25] ❌ 'Stop all' → fn:vol_up max

(Note: The parameter "max" is not a standard parameter for the `vol_up` function. Typically, `vol_
[ 8/25] ❌ 'Audio enable' → fn:vol_
[ 9/25] ❌ 'Play We Will Rock You' → fn:play_song "We Will Rock You"
[10/25] ❌ 'Play Wish You Were Here' → fn:play_song "Wish You Were Here"
[11/25] ❌ 'Forward' → I'm Phi, an AI language model, and I don't have the capability to control devices or systems. If you're trying to control a music player, you would
[12/25] ❌ 'Play Faithfully' → fn:play_song "Faithfully"
[13/25] ❌ 'Play Open Arms' → fn:play_song "Open Arms"
[14/25] ❌ 'Play my lounge playlist' → Sorry, I cannot help with that as I am an AI text-based assistant and do not have the capability to control devices or systems.
[15/25] ❌ 'Switch' → fn:vol_up
[16/25] ❌ 'Back' → I'm here to assist with information and guidance, but I don't have the capability to control external devices or systems. If you have questions or need assistance with information, feel free to
[17/25] ✅ 'Reverse' → fn:prev
[18/25] ❌ 'End' → fn:play_list "Morning Vibes"
fn:
[19/25] ❌ 'Finish it' → fn:next
[20/25] ❌ 'Pause now' → fn:vol_up
[21/25] ❌ 'Enough music' → I'm here to assist with information and guidance on a wide range of topics. If you have any questions or need assistance with something else, feel free to ask!
[22/25] ❌ 'Silent' → As an AI language model, I don't have the capability to control devices or functions. If you're trying to interact with a music player, please use the appropriate commands or functions as per
[23/25] ❌ 'No noise' → I understand you're requesting to stop the music. Since I'm text-based assistance, I can't directly control devices. To mute the music,
[24/25] ❌ 'What's your age?' → I'm Phi, an AI language model, so I don't have an age.
[25/25] ❌ 'Enable audio' → fn:

Summary: 2/25 correct (8.0%), avg 7.9s per request
```