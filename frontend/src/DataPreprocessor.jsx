import React, {useState, useEffect} from 'react'
import {Link, useLocation} from "react-router-dom";

const DataPreprocessor = () => {
    // const location = useLocation()
    // const videoName = location.state?.videoName
    // const {state} = props.location

    const videoName = "gs_training.mp4"

    const [extractionSuccess, setExtractionSuccess] = useState(null)
    const [extractionError, setExtractionError] = useState(null)

    useEffect(() => {
        // console.log(state)
        // console.log(videoName)
        const requestBody = {
            video_name: videoName,
            fps: 10
        };

        fetch('http://127.0.0.1:5000/api/extract-frames', {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody)
        })
            .then(res => res.json())
            .then(json => {
                if (json.errors) {
                    setExtractionError(json.errors.join(", "))
                } else {
                    setExtractionSuccess(json.frames_path)
                }
            })
            .catch(err => setExtractionError("An error occurred while video frames extraction."));
    }, [])


    return (
        <div>
            <h2>Extracting frames from the video...</h2>
            {/*<h3>[Optional: Preview statistics of frames/coordinates extracted, show a couple of skeletons]</h3>*/}

            <div>
                {extractionSuccess && <div style={{
                    color: "green",
                    marginTop: "10px"
                }}>Done! Video frames extraction successful.</div>}
                {extractionError && <div style={{
                    color: "red",
                    marginTop: "10px"
                }}>{extractionError}</div>}
            </div>
            <h2>Extracting 2d coordinates...</h2>
            <div style={{
                color: "green",
                marginTop: "10px"
            }}>Done! 2d coordinates extraction successful.
            </div>
            <Link to="/classification-results">Get classified images</Link>
        </div>
    )
}
export default DataPreprocessor