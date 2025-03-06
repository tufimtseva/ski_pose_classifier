import React, {useEffect, useState} from 'react'
// import { Link } from "react-router-dom";

const Home = () => {
    const [imageData, setData] = useState(null);
    const [expanded, setExpanded] = useState({
        left: false,
        middle: false,
        right: false
    });


    useEffect(() => {
        fetch('http://127.0.0.1:5000/ml-api/classify', {mode: 'cors'})
            .then(response => response.json())
            .then(json => setData(json))
            .catch(error => console.error(error));
    }, []);

    const toggleExpand = (folder) => {
        setExpanded((prevState) => ({
            ...prevState,
            [folder]: !prevState[folder],
        }));
    };


    return (
        <div>
            <h1>Ready!</h1>
            <h3>Here are the 3 folders with classified video frames my turn
                phase: left, middle and right</h3>
            {/*<div>*/}
            {/*    {imageData ? <pre>{JSON.stringify(imageData, null, 2)}</pre> : 'Loading...'}*/}
            {/*</div>*/}
            <div>
                {imageData && (
                    <>
                        {["left", "middle", "right"].map((folder) => (
                            <div key={folder} style={{marginBottom: "20px"}}>
                                <h2 onClick={() => toggleExpand(folder)}
                                    style={{cursor: "pointer"}}>
                                    {folder.charAt(0).toUpperCase() + folder.slice(1)} ({imageData[folder].length} images)
                                    {expanded[folder] ? " 🔽" : " ▶️"}
                                </h2>

                                {expanded[folder] && (
                                    <div style={{
                                        display: "flex",
                                        flexWrap: "wrap",
                                        gap: "10px"
                                    }}>
                                        {imageData[folder].map((imgBase64, index) => (
                                            <img
                                                key={index}
                                                src={`data:image/jpeg;base64,${imgBase64}`}
                                                alt={`${folder} ${index}`}
                                                width="150px"
                                                height="150px"
                                                style={{
                                                    objectFit: "cover",
                                                    borderRadius: "5px"
                                                }}
                                            />
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </>
                )}
            </div>
        </div>
    )
}
export default Home