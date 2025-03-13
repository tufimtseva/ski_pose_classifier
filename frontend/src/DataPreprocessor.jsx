import React, {useState, useEffect} from 'react'
import {Link, useLocation} from "react-router-dom";

const DataPreprocessor = () => {
    const location = useLocation()

    const [extractionSuccess, setExtractionSuccess] = useState(null) // todo rename to extractedFramesFolder ... frameExtractionError
    const [extractionError, setExtractionError] = useState(null)

    const [coordinatesSuccess, setCoordinatesSuccess] = useState(null)
    const [coordinatesFailure, setCoordinatesFailure] = useState(null)


    useEffect( () => {
        if (extractionSuccess === null)  {
            return
        }

        const requestBody = {
            frames_name: extractionSuccess
        };

        fetch('http://127.0.0.1:5000/api/extract-keypoints', {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody)
        })
            .then(res => res.json())
            .then(json => {
                if (json.errors) {
                    setCoordinatesFailure(json.errors.join(", "))
                } else {
                    setCoordinatesSuccess(json.json_folder_name)
                }
            })
            .catch(err => setExtractionError("An error occurred while 2d coordinates extraction."));

        }, [extractionSuccess])

    useEffect(() => {
        const videoName = location.state.videoName
        console.log(videoName)
        const requestBody = {
            video_name: videoName,
            fps: 10 // todo rollback to 10
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
                    setExtractionSuccess(json.frames_name)
                }
            })
            .catch(err => setExtractionError("An error occurred while video frames extraction."));
    }, [location])


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
            <div>
                {coordinatesSuccess && <div style={{
                    color: "green",
                    marginTop: "10px"
                }}>Done! 2d coordinates extraction successful.</div>}
                {coordinatesFailure && <div style={{
                    color: "red",
                    marginTop: "10px"
                }}>{coordinatesFailure}</div>}
            </div>
            <Link to="/classification-results" state={{ jsonFolderName: coordinatesSuccess}}>Get classified images</Link>
        </div>
    )
}
export default DataPreprocessor