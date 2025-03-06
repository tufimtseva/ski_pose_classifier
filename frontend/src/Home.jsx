import React from 'react'
 import { Link } from "react-router-dom";

const Home = () => {
    return(
        <div>
            <h1>Home</h1>
            <h3>Welcome to ski turn classifier!</h3>
            <h4>Guide</h4>
            <Link to="/load-video">Start</Link>
        </div>
    )
}
export  default Home