<p align="center">
    <img src="https://github.com/BigusD/ThugLife/blob/master/Pictures/BabyThugLife.png">
</p>

<h1 align="center"> ThugLife </h1>

Script that overlays sick thug life glasses on people in images. Script takes help from DLib bro's Facial Keypoint and Bounding Box detection models. Keypoints detected from these models are used as references to overlay the dope glasses. Check out the Pictures folder for more of Script dude's work.

## Contents
- [Getting Started](#Getting-Started)
    * [Environment](#Environment)
    * [Running the Script](#Running-the-Script)

## Getting Started
First, download `thuglife.yml`, `ThugLife.py`, `transparent.png`, and `shape_predictor_68_face_landmarks.dat`. Store all of these files in a directory you are comfortable with. Then using Anaconda, navigate to them. If you have stored your files within `\Downloads`, your feed into the prompt should look something like this.

```commandline
(base) C:\Users\[Name]> cd C:\Users\[Name]\Downloads 
```
Now, we need to create an environment with the needed dependencies for the script to function.

### Environment
There were quite a few dependencies, and it was a pain to get them all into one environment. To make things easier, I have exported the environment into `thuglife.yml` through which you can create environment with all the needed dependencies. With Anaconda this is simple.

```commandline
(base) C:\Users\[Name]\Downloads> conda env create -f thuglife.yml 
```

Then, make sure you activate the environment you have created.

```commandline
(base) C:\Users\[Name]\Downloads> conda activate thuglife
```

That's it. This will create an environment will all the needed dependencies to run the script.

### Running the Script
Now, we just need to tell the script where the image we want to ThugLife is. Obtain it's path and put this into the prompt as such. Don't forget to put the double quotes and make sure your image contains people. 

```commandline
(thuglife) C:\Users\[Name]\Downloads> python ThugLife.py -path "C:\Users\[Name]\Downloads\Baby.jpg"
```

Cue the Thug Life song! We have Thug Life'd some people with some dope glasses. To exit the image, simply press any key.
