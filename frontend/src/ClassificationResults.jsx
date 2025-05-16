import React, {useEffect, useState} from 'react';
import {useLocation, Link} from "react-router-dom";


const Home = () => {
    const location = useLocation();

    const [imageData, setImageData] = useState(null);
    const [probsData, setProbsData] = useState(null);
    const [jsonFolderName, setJsonFolderName] = useState(null);
    const [expandedFolders, setExpandedFolders] = useState({
        left: false,
        middle: false,
        right: false,
    });
    const [expandedGroups, setExpandedGroups] = useState({});

    useEffect(() => {
        const jsonFolderName = location.state.image_folder_name;
        setJsonFolderName(jsonFolderName);
        const requestBody = {image_folder_name: jsonFolderName};

        fetch('http://127.0.0.1:5000/ml-api/classify-images', {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(requestBody)
        })
            .then(res => res.json())
            .then(json => {
                setImageData(json.images);
                console.log("probs ", json.probabilities)
                setProbsData(json.probabilities);
            })
            .catch(err => {
                console.error("Classification failed:", err.message);
            });
    }, [location]);
    // todo revert
    //
    //
    // useEffect(() => {
    //     const jsonFolderName = "gs_run_fps3";
    //     const requestBody = { json_folder_name: jsonFolderName };
    //
    //     fetch('http://127.0.0.1:5000/ml-api/classify', {
    //         method: "POST",
    //         headers: { "Content-Type": "application/json" },
    //         body: JSON.stringify(requestBody)
    //     })
    //         .then(res => res.json())
    //         .then(json => {
    //             setImageData(json.images);
    //             setProbsData(json.probabilities);
    //         })
    //         .catch(console.error);
    // }, []);


    const toggleFolder = (folder) => {
        setExpandedFolders(prev => ({...prev, [folder]: !prev[folder]}));
    };

    const toggleGroup = (folder, groupIndex) => {
        const key = `${folder}_${groupIndex}`;
        setExpandedGroups(prev => ({...prev, [key]: !prev[key]}));
    };

    return (
        <div>
            <h1>Ready!</h1>
            <h3>Here are the 3 top-level folders with classified video frames by
                turn phase:</h3>

            {imageData && ["left", "middle", "right"].map(folder => (
                <div key={folder} style={{marginBottom: "30px"}}>
                    <h2
                        onClick={() => toggleFolder(folder)}
                        style={{cursor: "pointer", marginBottom: "10px"}}
                    >
                        {folder.charAt(0).toUpperCase() + folder.slice(1)} ({imageData[folder].length} groups)
                        {expandedFolders[folder] ? " 🔽" : " ▶️"}
                    </h2>

                    {expandedFolders[folder] && (
                        <div style={{paddingLeft: "20px"}}>
                            {imageData[folder].map((group, groupIdx) => {
                                const groupKey = `${folder}_${groupIdx}`;
                                return (
                                    <div key={groupKey}
                                         style={{marginBottom: "20px"}}>
                                        <h4
                                            onClick={() => toggleGroup(folder, groupIdx)}
                                            style={{
                                                cursor: "pointer",
                                                marginBottom: "5px"
                                            }}
                                        >
                                            {folder} {groupIdx + 1} ({group.length} frames)
                                            {expandedGroups[groupKey] ? " 🔽" : " ▶️"}
                                        </h4>

                                        {expandedGroups[groupKey] && (
                                            <div style={{
                                                display: "flex",
                                                flexWrap: "wrap",
                                                gap: "10px",
                                                paddingLeft: "20px"
                                            }}>
                                                {group.map((imgBase64, imgIdx) => (
                                                    <img
                                                        key={imgIdx}
                                                        src={`data:image/jpeg;base64,${imgBase64}`}
                                                        alt={`${folder}_${groupIdx}_${imgIdx}`}
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
                                );
                            })}
                        </div>
                    )}
                </div>
            ))}
            {probsData && (
                <Link
                    to="/probabilities"
                    state={{probabilities: probsData, jsonFolderName: jsonFolderName}}
                >
                    Show Probabilities
                </Link>
            )}

        </div>
    );
};

export default Home;
