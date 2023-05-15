# kmdx-net Music Source Separation


# Submission
My submission for the [AIcrowd music demixing challenge](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23/leaderboards)
## Submission Summary

* MDX Leaderboard C
	* SDR scores from the AIcrowd/Sony hidden dataset:
	  |  SDR_song | SDR_bass | SDR_drums | SDR_other | SDR_vocals |
	  | :------: | :------: | :-------: | :-------: | :--------: |
	  |   9.181   |   10.056   |   9.465   |   6.804    |    10.398    |

## Model Summary

* Models
  * Ensemble of [Demucs](https://github.com/facebookresearch/demucs) and [Kuielab mdx-net](https://github.com/kuielab/mdx-net)
* Demucs models (all trained by meta)
  * 4 x htdemucs_ft
  * demucs_mmi
  * a1d90b5c
* Kuielab mdx-net models (all trained by me on a single 16gb T4 GPU)
  * vocals.onnx (trained with batchnorm2d, the vocals model is an onnx file because it was trained before i realized pytorch models were faster and i deleted the checkpoint)
  * bass.pt (trained with groupnorm num_groups=4)
  * other.pt (trained with groupnorm num_groups=2)
  * drums.pt (trained with groupnorm num_groups=2)
* Things i did to boost SDR
  * modify the input mixtures for each model
    * input mixture for htdemucs_ft vocals = The original mixture
    * input mixture for htdemucs_ft drums = Original mixture - output of htdemucs_ft vocals
    * input mixture for htdemucs_ft bass = Original mixture - output of htdemucs_ft vocals + drums
    * input mixture for bass, drums and other.pt = Original mixture - output of vocals.onnx
  * Using original mixture - output of htdemucs_ft vocals + drums + bass as the other stem scored higher than using the actual other stem model.
  * Blending the model outputs
    * First i blended the demucs model outputs together, then blended the demucs outputs with the mdx-net model outputs
    * [0.08, 0.08, 0.4, 0.88] was the final blend value between demucs and mdx-net (0 would mean 100% demucs was used and 1 would mean 100% mdx-net) in the order of drums,bass, other and vocals.   
## Download the models from my google drive

https://drive.google.com/drive/folders/1ugGoKCsnRdeFjez89o4Ut_bf6wUv85wy?usp=sharing

## Thanks to
https://github.com/rlaalstjr47

https://github.com/ws-choi

https://github.com/facebookresearch/demucs

https://github.com/Anjok07

https://github.com/aufr33

https://github.com/ZFTurbo


