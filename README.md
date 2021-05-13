A Pytorch Implementation for DA comparaison

## Introduction
Ce gitHhub a été créé pour tester différentes techniques de Knowledge Distilation (KD) pour le Faster-R-CNN. Nous avons testé la technique [Overhaul of Feature Distillation](https://github.com/clovaai/overhaul-distillation) que nous avons modifié pour qu'elle s'applique au Faster-R-CNN et la technique [Fine-grained Feature Imitation](https://github.com/twangnh/Distilling-Object-Detectors)
Pour l'environnement, il faut suivre ce git : [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0). Dans ce projet nous utilisons CUDA 10.2 et PyTorch 1.1

## Le format des données
Toutes les données sont **au format PASCAL_VOC**.  
Si vous voulez utiliser ce code pour votre propre dataset, il faut arranger votre dataset au format Pascal, créer un fichier dans ```lib/datasets/```, l'ajouter sur  ```lib/datasets/factory.py```. Et enfin modifier ```lib/model/utils/parser_func.py```.


## Modèle
### Modèle pré-entrainé
Dans nos expérimentations, nous utilisons des modèles pré-entrainés sur ImageNet

### Faites attention
Les modeles que nous utilisons sont pré-entrainés avec Pytorch, donc les images sont modifiées pour qu'elles soient en RGB et les pixels entre \[0-1].
Si vous voulez utiliser des modèles Caffe, il faudra modifier le code

