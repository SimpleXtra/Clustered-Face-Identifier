# Dataset

This dataset is used to train the face detection (and tracking) model. The dataset consists of **32.839** images contaning face(s) along with their respective bounding boxes, with the following distribution:

- 26.266 images used for the *training* dataset.
- 6.573 images used for the *validation* dataset.

All data are sourced from Kaggle. The links are available at the bottom of this page.

The raw ZIP files from Kaggle are stored in `data/raw/`, while the combined and formatted dataset is stored in `data/processed/`. Images are stored in `data/processed/images/` in JPG format, and the labels (bounding boxes) are stored in `data/processed/labels/`.

## Data Format

The formatted dataset follows the [Ultralytics YOLO format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format).

Since the model only detects faces, the `dataset.yaml` definition only uses one class (`0`) to represent faces.

### Preprocessing

> All preprocessing that tinkers with the dataset structure are addressed in `notebooks/dataset_preprocessing.ipynb`.

One of the [datasets](https://www.kaggle.com/datasets/sudhanshu2198/face-detection-dataset) uses the `xywh` format to represent the bounding boxes, where the `xy` is the top-left corner of the face, and `wh` is the width and height of the bounding box. However, YOLO expects a normalized `xywh`, where the `xy` is the center point of the face instead. Therefore, all labels from this dataset source must be converted to a normalized `xywh` format first.

The same dataset also originally contains a *test* set. However, since no labels exist for the *test* set, it is ignored.

## Sources

- [Face Detection Dataset (Sudhanshu Rastogi)](https://www.kaggle.com/datasets/sudhanshu2198/face-detection-dataset)

- [Face-Detection-Dataset (Fares Elmenshawii)](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)