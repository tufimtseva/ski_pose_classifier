import React from 'react'
 import { Link } from "react-router-dom";

const VideoLoader = () => {
    return(
        <div>
            <h1>The first step: upload your video</h1>
            <h3>Upload the video of your ski run below. The video frames will be extracted with the default fps (Frames Per Second) = 10. You can also set a custom fps by yourself below.</h3>
            <h4>Choose a custom fps</h4>
            <h4>Click here to upload your video</h4>
            <Link to='/preprocess-data'>Continue</Link>
        </div>
    )
}
export  default VideoLoader