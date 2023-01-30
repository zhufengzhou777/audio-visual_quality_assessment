# audio-visual quality assessment
## 1.obtain features extracted by ViT
get the pretrained ViT model by rwightman\
link:https://github.com/rwightman/pytorch-image-models/
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