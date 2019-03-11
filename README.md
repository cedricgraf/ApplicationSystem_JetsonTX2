# ApplicationSystem_JetsonTX2
Academic Project for detection and localisation of objectes by deep learning


## Modèles TensorFlow / TensorRT sur Jetson
Ce référentiel a été créé à partir du répertoire [tf_trt_models](https://github.com/NVIDIA-Jetson/tf_trt_models) de NVIDIA. Il contient des instructions pour optimiser les modèles TensorFlow avec TensorRT, ainsi que des scripts de test / démonstration. Les modèles proviennent du répertoire de modèles TensorFlow [TensorFlow models repository](https://github.com/tensorflow/models). Ce répertoire se concentre principalement sur les modèles de détection d'objets.

*[Installer] (#ins)
*[Détection d'objets] (#do)
  *[Modèles] (#mo)
  *[Détection d'objets en temps réel avec les modèles optimisés TensorRT](#rt)
*Application du modèle de détecteur à main

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
    
