# AirCanvas

AirCanvas is a Python script that transforms your webcam feed into an interactive canvas. Draw colorful lines with your fingers and use the color palette buttons to switch between colors.

## Prerequisites
Ensure you have the required packages installed:
```bash
pip install opencv-python
pip install mediapipe
```
## How to Use<br>
1) Clone the repository.
2) Navigate to the project directory.
3) Run the script:
```bash
python air_canvas.py
```
4) Enjoy drawing on the canvas with your webcam!

## Controls
- Use your fingers to draw on the canvas.
- Press the CLEAR button to clear the canvas.
- Use the color palette buttons (BLUE, GREEN, RED, YELLOW) to switch between colors.
 
### Troubleshooting
- OpenCV Installation Issue:
- If there are issues with OpenCV installation, run the following command:
```bash
pip install opencv-python
```
## Window Resizing Error:
If you encounter a window resizing error, make sure your OpenCV version is up-to-date.

## Null Pointer Errors:
If you face Null Pointer errors when resizing or setting window properties, try checking if the window name matches correctly. In some cases, errors might be due to inconsistencies in window names.

## Draw Landmarks Error:
If you encounter unresolved reference 'mpDraw', 'mpHands', ensure that the mpHands and mpDraw objects are defined globally.

## Known Issues
The code might contain a few bugs. Your patience is appreciated while I work on improving the script.

For any other issues or bug reports, feel free to raise an issue.

## Happy drawing! 🎨



