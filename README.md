**SkiPoseClassifier** is designed for a *real-time image classification* of athletes by turn phases for *alpine skiing* technique analysis.

For keypoint detection of athletes' poses, **SkiPoseClassifier** utilizes [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), a multi-person system to jointly detect human body, hand, facial, and foot keypoints (in total 135 keypoints) on single images.

### Installation

1. The first step is to clone the SkiPoseClassifier repository.
```bash
git clone git@github.com:tufimtseva/ski_pose_classifier.git
```
2. To start backend server, from the **root directory**, run:

```bash
cd backend
flask run
```
3. To start frontend server, go back to the **root directory** and run:
```bash
cd frontend
npm install  
npm run dev 
```
Run ```npm install``` only once for the first time to install dependencies.

