import React, {useState} from 'react'
import {Link} from "react-router-dom";

const VideoLoader = () => {
    const [file, setFile] = useState(null)
    const [uploadSuccess, setUploadSuccess] = useState(null)
    const [uploadError, setUploadError] = useState(null)


    // const marks = [
    //     {
    //         value: 2,
    //         label: '2 fps',
    //     },
    //     {
    //         value: 5,
    //         label: '5 fps',
    //     },
    //     {
    //         value: 8,
    //         label: '8 fps',
    //     },
    //     {
    //         value: 10,
    //         label: '10 fps',
    //     }
    // ];


    const handleUploading = () => {
        const formData = new FormData()
        formData.append('file', file)

        fetch('http://127.0.0.1:5000/api/video-upload', {
            method: "POST",
            body: formData
        })
            .then(res => res.json())
            .then(json => {
                if (json.errors) {
                    setUploadError(json.errors.join(", "))
                } else {
                    setUploadSuccess(json.filename)
                }
            })
            .catch(err => setUploadSuccess("An error occurred while video uploading."));
    }

    return (
        <div>
            <h1>The first step: upload your video</h1>
            <h3>Upload the video of your ski run below. The video frames
                will be
                extracted with the default fps (Frames Per Second) = 10.</h3>
            <h4>Click here to upload your video</h4>
            <div>
                <input type="file" onChange={(event) => {
                    setFile(event.target.files[0])
                }}/>
                {file && <button onClick={handleUploading}>Upload</button>}
            </div>
            <div>
                {uploadSuccess && <div style={{
                    color: "green",
                    marginTop: "10px"
                }}>Uploaded: {uploadSuccess}</div>}
                {uploadError && <div style={{
                    color: "red",
                    marginTop: "10px"
                }}>{uploadError}</div>}
            </div>

            {uploadSuccess && <Link to="/preprocess-data" state={{ videoName: uploadSuccess }}>Continue</Link>}
        </div>
    )
}
export default VideoLoader