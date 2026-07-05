# Detekcija pozicije šake i broja podignutih prstiju

## Opis projekta

Projekat ima za cilj kreiranje modela mašinskog učenja koji će na slici prepoznati poziciju šake i broj podignutih prstiju.

Projekat je podeljen u 3 celine:

1. Klasifikacija uz pomoć klasičnih modela i konvolutivnih neuronskih mreža [detaljnije](./src/classic_models/README.md)
2. Segmentacija instanci modelom YOLO familije [detaljnije](./src/yolo_instance_segmentation/README.md)
3. Detekcija šake i klasifikacija modelom YOLO familije [detaljnije](./src/yolo_detection/README.md)

## Skupovi podataka

Za treniranje modela korišćeni su sledeći kupovi podataka:

1. [Fingers numbers](https://universe.roboflow.com/hands-rirpj/fingers-numbers)
2. [EGO Hands](https://www.kaggle.com/datasets/himaniishah/egohands)
3. [Hands segmented](https://www.kaggle.com/datasets/thebestpatate/11k-hands-segmented-with-sam3-yolo-format)
4. [Podskup skupa HaGRID](https://www.kaggle.com/datasets/markolazarevi/hagrid-fingers-yolo)

Detaljni opisi skupova podataka i način njihovog korišćenja su objašnjeni u README fajlovima celina projekta u kojima su korišćeni.

## Članovi tima

- Milan Bodo 1026/2025
- Marko Koprivica 1010/2025
- Marko Lazarević 1005/2025
