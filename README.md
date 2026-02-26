# Horse Video Auto Clipper

## Mission / Description

This desktop application automatically identify the walking direction of a horse in yearling parade videos, detect the frame where the horse is most perpendicular to the camera while walking left-to-right, and extract standardised clips (±2 seconds) around that moment.

![teaser image](teaser.png)


Sample horse videos can be downloaded from [here](https://drive.google.com/drive/folders/1vpvZqH313YYjHVr6MA4gc0j96oMm-UyA?usp=sharing).

## Creating Env in Win 11

```
conda create -n HorseVideoAutoClipper_Env python=3.10
conda activate HorseVideoAutoClipper_Env

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Running Source Code

```
python main.py
```


## Making Single Exe file

```
pyinstaller main.py --add-data "configs/bytetrack.yaml;configs" --add-data "models/yolov8m.pt;models"
```
