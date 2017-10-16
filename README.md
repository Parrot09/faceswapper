# Face Swapper

![Alt text](./bill-steve.jpg?raw=true "Bill Gates and Steve Jobs" | width=100)
![Alt text](./trump-putin.jpg?raw=true "Donald Trump and Vladimir Putin" | width=100)

## About

Python program to swap faces between two images. Requires a source image and a destination image. The destination image can contain multiple faces. 

Created for HackNY 2017

## How it works

Uses the Clarifai face recognition model to detect faces in both images. Then uses dlib to detect facial landmarks in the source image. Extracts only the landmarks by creating a convex hull around the landmark points using OpenCV. Applies these landmarks on the destination image.

## Dependencies

- [Clarifai](https://www.clarifai.com/)
- [dlib](http://dlib.net/)
- [OpenCV](https://opencv.org/)

## Running

Make sure all the dependencies are met and register for a Clarifai API key.

Change the following in `faceswapper.py`

```
app = ClarifaiApp(api_key='YOUR_API_KEY')
```

Use the API key here and then change

```
source_file = 'bill-gates.jpg'
target_file = 'steve-jobs.jpg'
```

to try this with other images
