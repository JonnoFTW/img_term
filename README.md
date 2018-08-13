Image 2 ANSI
=
Converts an image to a representation in ANSI

Installation
-
1. Copy the repository
2. Run:
```bash
pip install opencv-python numba numpy
```
Usage
-

```
usage: img_term.py [-h] [-img IMG] [-width WIDTH] [-cam CAM] [-col {4,8,24}]

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
./img_term.py -img dog.jpg 
```
Render your video capture device to terminal:
```bash
./img_term.py
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

Or simply stream your USB camera to terminal!:

![Screen](/RemarkablePalatableKob.webm)