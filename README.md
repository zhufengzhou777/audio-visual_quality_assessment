# audio-visual quality assessment
## 1.obtain features extracted by ViT
get the pretrained ViT model by rwightman\,save them at ./pretrained_models
link:https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth

https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth

````
python feature_extractor/feature_extractor.py --input (path) --output (path)
````
## 2.get the train list and the test list
````
python getlist.py 
````
## 3.train the AVQA model
````
python train.py --epoch (epoch) --learning_rate (learning_rate) --batch_size (batch_size)
````
## 4.get result
obtain trainlog in ./log/avqa.log\
obtain plot in ./log/avqa.png