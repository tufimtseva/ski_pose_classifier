import React from 'react'
 import { Link } from "react-router-dom";

const Home = () => {
    return(
        <div>
            <h1>Home</h1>
            <h3>Welcome to ski turn classifier!</h3>
            <Link>Guide</Link>
            <Link to="/load-video">Start</Link>
        </div>
    )
}
export  default Home