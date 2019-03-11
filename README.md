# ApplicationSystem_JetsonTX2
Academic Project for detection and localisation of objectes by deep learning


## Modèles TensorFlow / TensorRT sur Jetson
Ce référentiel a été créé à partir du répertoire [tf_trt_models](https://github.com/NVIDIA-Jetson/tf_trt_models) de NVIDIA. Il contient des instructions pour optimiser les modèles TensorFlow avec TensorRT, ainsi que des scripts de test / démonstration. Les modèles proviennent du répertoire de modèles TensorFlow [TensorFlow models repository](https://github.com/tensorflow/models). Ce répertoire se concentre principalement sur les modèles de détection d'objets.

* [Installer](#ins)
* [Détection d'objets](#do)
  * [Modèles](#mo)
  * [Détection d'objets en temps réel avec les modèles optimisés TensorRT](#rt)
  * [Comment ça marche](#cm)
   * [Classification d'image](#ci)
   * [Détection](#d)
* [Application du modèle de détecteur à main](#main)

<a name ="ins"></a>
### Installer

1. Faites flasher le système cible Jetson TX2 avec JetPack-3.2.1 (TensorRT 3.0 GA inclus) ou JetPack 3.3 (TensorRT 4.0 GA).

2. Installez OpenCV 3.4.x sur Jetson. Référence: Reference: [How to Install OpenCV (3.4.0) on Jetson TX2](https://jkjung-avt.github.io/opencv3-on-tx2/).

3. Téléchargez et installez TensorFlow 1.8.0 (avec le support TensorRT). Notez que TensorFlow 1.9.0, 1.10.0 et 1.10.1 ne fonctionnent pas bien sur Jetson TX2. TensorFlow 1.8.0 est fortement recommandé au moment de la rédaction de cet article si vous souhaitez utiliser TF-TRT sur Jetson TX2. Notez également que python3 a été utilisé pour tous les travaux de test et de développement de l'auteur.

Téléchargez **[ce pip](https://nvidia.app.box.com/v/TF180-Py35-wTRT)** si vous utilisez JetPack-3.2.1. Sinon, téléchargez **[ce pip](https://drive.google.com/open?id=1bAUNe26fKgGXuJiZYs1eT2ig8SCj2gW-)** si vous utilisez JetPack-3.3.
   ```
    $ sudo pip3 installez tensorflow-1.8.0-cp35-cp35m-linux_aarch64.whl
   ```
4. Cloner ce répertoire. (Utilisez ce référentiel à la place du répertoire tf_trt_models d'origine de NVIDIA, si vous souhaitez exécuter le script décrit ci-dessous.)
   ```
    $ cd ~ / projet
    $ git clone --recursive https://github.com/cedricgraf/ApplicationSystem_JetsonTX2/edit/master/README.md
    $ cd tf_trt_models
   ```
5. Exécutez le script d'installation.
    ```
     $ ./install.sh
    ```
<a name="do"></a>    
### Détection d'objets
Veuillez vous reporter à l'original de  [NVIDIA-Jetson/tf_trt_models](https://github.com/NVIDIA-Jetson/tf_trt_models) pour des extraits de code qui montrent comment télécharger des modèles de détection d'objet non entraînés, comment créer un graphique TensorFlow et comment optimiser les modèles avec TensorRT.

<a name="mo"></a>
### Modèles

Notez que les temps de référence ont été rassemblés après que le Jetson TX2 ait été placé en mode MAX-N. Pour définir TX2 en mode MAX-N, exécutez les commandes suivantes dans un terminal:
 ``` 
  $ sudo nvpmodel -m 0
  $ sudo ~ / jetson_clocks.sh
 ```
<a name="do"></a>
### Détection d'objets en temps réel avec les modèles optimisés TensorRT
Le script camera_tf_trt.py prend en charge les entrées vidéo provenant de l’une des sources suivantes: (1) un fichier vidéo, par exemple, mp4, (2) un fichier image, par exemple, jpg ou png, (3) un flux RTSP provenant d’un IP CAM, (4). ) une webcam USB, (5) la caméra embarquée Jetson. Consultez le message d'aide sur la façon d'appeler le script avec une source vidéo spécifique.

```
$ python3 camera_tf_trt.py --help
usage: camera_tf_trt.py [-h] [--file] [--image] [--filename FILENAME] [--rtsp]
                        [--uri RTSP_URI] [--latency RTSP_LATENCY] [--usb]
                        [--vid VIDEO_DEV] [--width IMAGE_WIDTH]
                        [--height IMAGE_HEIGHT] [--model MODEL] [--build]
                        [--tensorboard] [--labelmap LABELMAP_FILE]
                        [--num-classes NUM_CLASSES] [--confidence CONF_TH]

This script captures and displays live camera video, and does real-time object
detection with TF-TRT model on Jetson TX2/TX1

optional arguments:
  -h, --help            show this help message and exit
  --file                use a video file as input (remember to also set
                        --filename)
  --image               use an image file as input (remember to also set
                        --filename)
  --filename FILENAME   video file name, e.g. test.mp4
  --rtsp                use IP CAM (remember to also set --uri)
  --uri RTSP_URI        RTSP URI, e.g. rtsp://192.168.1.64:554
  --latency RTSP_LATENCY
                        latency in ms for RTSP [200]
  --usb                 use USB webcam (remember to also set --vid)
  --vid VIDEO_DEV       device # of USB webcam (/dev/video?) [1]
  --width IMAGE_WIDTH   image width [1280]
  --height IMAGE_HEIGHT
                        image height [720]
  --model MODEL         tf-trt object detecion model [ssd_inception_v2_coco]
  --build               re-build TRT pb file (instead of usingthe previously
                        built version)
  --tensorboard         write optimized graph summary to TensorBoard
  --labelmap LABELMAP_FILE
                        [third_party/models/research/object_detection/data/msc
                        oco_label_map.pbtxt]
  --num-classes NUM_CLASSES
                        number of object classes [90]
  --confidence CONF_TH  confidence threshold [0.3]
```

L'option `--model` peut uniquement être définie sur` ssd_inception_v2_coco` (par défaut) ou `ssd_mobilenet_v1` maintenant. Il serait probablement étendu pour prendre en charge davantage de modèles de détection d'objets à l'avenir. L'option `--build` ne doit être effectuée qu'une seule fois pour chaque modèle de détection d'objet. Le graphe optimisé TensorRT serait sauvegardé / mis en cache dans un fichier protobuf, de sorte que les appels ultérieurs du script puissent charger le graphe en cache directement sans avoir à repasser par le processus d'optimisation.


Exemple n ° 1: Compiler le modèle 'ssd_mobilenet_v1_coco' optimisé par TensorRT et exécutez la détection d'objet en temps réel
```
$ python3 camera_tf_trt.py --usb --model ssd_mobilenet_v1_coco --build
```
Exemple n ° 2: vérifiez le modèle optimisé 'ssd_mobilenet_v1_coco' avec la photo «huskies.jpg» d'origine de NVIDIA.
```
$ python3 camera_tf_trt.py --image --filename examples/detection/data/huskies.jpg --model ssd_mobilenet_v1_coco
```

<p>
<img src="data/huskies_detected.png" alt="MobileNet V1 SSD detection result on huskies.jpg" height="300px"/>
</p>

<a name="cm"></a>
### Comment ça marche

<a name="ci"></a>
#### Classification d'image
<img src="data/classification_graphic.jpg" alt="classification" height="300px"/>


#### Modèles

| Model | Input Size | TF-TRT TX2 | TF TX2 |
|:------|:----------:|-----------:|-------:|
| inception_v1 | 224x224 | 7.36ms | 22.9ms |
| inception_v2 | 224x224 | 9.08ms | 31.8ms |
| inception_v3 | 299x299 | 20.7ms | 74.3ms |
| inception_v4 | 299x299 | 38.5ms | 129ms  |
| mobilenet_v1_0p25_128 | 128x128 | 3.72ms | 7.99ms |
| mobilenet_v1_0p5_160 | 160x160 | 4.47ms | 8.69ms |
| mobilenet_v1_1p0_224 | 224x224 | 11.1ms | 17.3ms |

**TF** - Original TensorFlow graph (FP32)

**TF-TRT** - TensorRT optimized graph (FP16)

#### Télécharger les modèles pre-entrainer

Exemple modèle inception_v2 de google
```python
from tf_trt_models.classification import download_classification_checkpoint

checkpoint_path = download_classification_checkpoint('inception_v2')
```
Cliquer sur ce lien pour télécharger manuellement les modèles [ici](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained).


#### Construire TensorRT

```python
from tf_trt_models.classification import build_classification_graph

frozen_graph, input_names, output_names = build_classification_graph(
    model='inception_v2',
    checkpoint=checkpoint_path,
    num_classes=1001
)
```

### Optimiser avec TensorRT

```python
import tensorflow.contrib.tensorrt as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)
```
Détection 
----------------

<img src="data/detection_graphic.jpg" alt="detection" height="300px"/>


#### Models

| Model | Input Size | TF-TRT TX2 | TF TX2 |
|:------|:----------:|-----------:|-------:|
| ssd_mobilenet_v1_coco | 300x300 | 50.5ms | 72.9ms |
| ssd_inception_v2_coco | 300x300 | 54.4ms | 132ms  |

**TF** - Original TensorFlow graph (FP32)

**TF-TRT** - TensorRT optimized graph (FP16)

#### Télécharger les modèles pre-entrainer

```python
from tf_trt_models.detection import download_detection_model

config_path, checkpoint_path = download_detection_model('ssd_inception_v2_coco')
```
Cliquer sur ce lien pour télécharger manuellement les modèles [ici](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

#### Construire TensorRT

```python
from tf_trt_models.detection import build_detection_graph

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path
)
```

#### Optimiser avec TensorRT

```python
import tensorflow.contrib.tensorrt as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)
```

<a name="main"></a>
### Application du modèle de détecteur à main
* [Training a Hand Detector with TensorFlow Object Detection API](https://jkjung-avt.github.io/hand-detection-tutorial/)
* [Deploying the Hand Detector onto Jetson TX2](https://jkjung-avt.github.io/hand-detection-on-tx2/)

Une fois que vous avez formé votre propre détecteur de mains avec l’un des modèles suivants, vous pourrez l’optimiser avec TF-TRT et l’exécuter sur TX2.
```
ssd_mobilenet_v1_egohands
ssd_mobilenet_v2_egohands
ssdlite_mobilenet_v2_egohands
ssd_inception_v2_egohands
faster_rcnn_resnet50_egohands
faster_rcnn_resnet101_egohands
faster_rcnn_inception_v2_egohands
```

Veillez à copier vos fichiers de point de contrôle de modèle formés dans le dossier `data / xxx_egohands /` correspondant. Disons que vous avez fait cela pour `ssd_mobilenet_v1_egohand`. Ensuite, vous pouvez optimiser le modèle et le tester avec une image comme celle-ci:
```shell
$ python3 camera_tf_trt.py --image \
                           --filename jk-son-hands.jpg \
                           --model ssd_mobilenet_v1_egohands \
                           --labelmap data/egohands_label_map.pbtxt \
                           --num-classes 1 \
                           --build
```

<p>
<img src="https://jkjung-avt.github.io/assets/2018-09-25-hand-detection-on-tx2/son-hands-detected.png" alt="JK's son's hands" height="300px"/>
</p>

