# Horse Video Auto Clipper

## Mission / Description

This desktop application automatically identify the walking direction of a horse in yearling parade videos, detect the frame where the horse is most perpendicular to the camera while walking left-to-right, and extract standardised clips (±2 seconds) around that moment.

![teaser image](teaser.png)


Sample horse videos can be downloaded from [here](https://drive.google.com/drive/folders/1vpvZqH313YYjHVr6MA4gc0j96oMm-UyA?usp=sharing).

## Running Source Code

```
pip install -r requirements.txt
python main.py
```


## Making Single Exe file

```
pyinstaller main.py --add-data "configs/bytetrack.yaml;configs" --add-data "models/yolov8m.pt;models"
```
