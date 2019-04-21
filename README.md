Image 2 ANSI
=
Converts an image to a representation in ANSI

Installation
-
1. 
```bash
pip install img_term
```

Usage
-

```
usage: img_term [-h] [-img IMG] [-width WIDTH] [-cam CAM] [-col {4,8,24}]

Display image to terminal

optional arguments:
  -h, --help     show this help message and exit
  -img IMG       Image file to display
  -width WIDTH   Character width of output
  -cam CAM       Show camera, this is the default
  -col {4,8,24}  Colour scheme to use
```

Display an image in terminal 
```bash
img_term -img dog.jpg 
```

Render your video capture device to terminal:
```bash
img_term
```

Select a colour palette:

```bash
./img_term.py -img lena.jpg -col 24
```

Example
-
Go from this:

![Dog](/dog.jpg)

To this:

![Screenshot](/screenshot.png)


Or this

![Lenna](/lena.jpg)

To this:

![LenaPixel](/screenshot2.png)

Or even render a video:

```bash
./img_term.py -col 24 -vid unrealset.mkv -width 78
```

https://gfycat.com/IdolizedSomeGemsbuck

Or simply stream your USB camera to terminal!:

https://gfycat.com/RemarkablePalatableKob