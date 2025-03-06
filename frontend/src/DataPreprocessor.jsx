import React from 'react'
 import { Link } from "react-router-dom";
const DataPreprocessor = () => {


    return(
        <div>
            <h2>Extracting frames from the video... Done!</h2>
            <h2>Extracting 2d coordinates... Done!</h2>
            {/*<h3>[Optional: Preview statistics of frames/coordinates extracted, show a couple of skeletons]</h3>*/}
            <Link to="/classification-results">Get classified images</Link>
        </div>
    )
}
export  default DataPreprocessor