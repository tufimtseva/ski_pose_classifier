import { useState } from 'react'
import { Route, Routes } from 'react-router-dom';
import './App.css'

import Home from "./Home.jsx";
import VideoLoader from "./VideoLoader.jsx";
import ClassificationResults from "./ClassificationResults.jsx";
import DataPreprocessor from "./DataPreprocessor.jsx";
function App() {

  return (
    <>
      <div>
        <Routes>
            {/*todo check route naming*/}
            <Route path='/' element={<Home/>}></Route>
            <Route path='/load-video' element={<VideoLoader/>}></Route>
            <Route path='/classification-results' element={<ClassificationResults/>}></Route>
            <Route path='/preprocess-data' element={<DataPreprocessor/>}></Route>
        </Routes>
      </div>
    </>
  )
}

export default App
