// src/components/ProbabilityPlotPage.js
import React, {useEffect, useState} from 'react';
import {Link, useLocation} from 'react-router-dom';
import {
    ResponsiveContainer,
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    CartesianGrid,
    Label
} from 'recharts';

const colors = {
    Left: '#1f77b4', // blue
    Middle: '#ff7f0e', // orange
    Right: '#2ca02c', // green
};

const getTimeTicks = (data) => {
    return data
        .map(d => d.time)
        .filter((t, i, arr) => parseFloat(t) % 0.5 === 0);
};


const transformProbabilities = (probs, fps) => {
    return probs.map((frameProbs, index) => ({
        frame: index,
        time: (index / fps).toFixed(2),
        Left: frameProbs[0],
        Middle: frameProbs[1],
        Right: frameProbs[2],
    }));
};

const ProbabilityPlotPage = () => {
    const location = useLocation();
    const probabilities = location.state.probabilities;
    const jsonFolderName = location.state.jsonFolderName;
    const [data, setData] = useState([]);
    const fps = 10;

    useEffect(() => {
        if (probabilities) {
            console.log("probs: ", probabilities)
            const transformed = transformProbabilities(probabilities, fps);
            setData(transformed);
        }
    }, [probabilities]);

    return (
        <div style={{width: 400}}>
            <h2>Phase Probabilities over Time</h2>
            <ResponsiveContainer width="100%" height={600}>
                <LineChart
                    layout="vertical"
                    data={data}
                    margin={{top: 20, right: 50, left: 50, bottom: 40}}
                >
                    <CartesianGrid strokeDasharray="3 3"/>
                    <XAxis type="number" domain={[0, 1]}>
                        <Label value="Probability" position="insideBottom"
                               offset={-5}/>
                    </XAxis>
                    <YAxis type="category" dataKey="time" interval={0} ticks={getTimeTicks(data)}>
                        <Label value="Time (s)" angle={-90}
                               position="insideLeft"/>
                    </YAxis>
                    <Tooltip/>
                    <Legend wrapperStyle={{left: 80, top: 580}}/>
                    <Line type="monotone" dataKey="Left" stroke={colors.Left}
                          strokeWidth={2}/>
                    <Line type="monotone" dataKey="Middle"
                          stroke={colors.Middle} strokeWidth={2}/>
                    <Line type="monotone" dataKey="Right" stroke={colors.Right}
                          strokeWidth={2}/>
                </LineChart>
            </ResponsiveContainer>
            <Link
                to="/classification-results"
                state={{jsonFolderName: jsonFolderName}}
            >
                Go back
            </Link>
        </div>
    );
};

export default ProbabilityPlotPage;
