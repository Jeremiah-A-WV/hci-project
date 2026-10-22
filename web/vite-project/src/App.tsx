import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [telemetry, setTelemetry] = useState({
    status: 'Disconnected',
    roll: 0,
    throttle: 0,
    battery: 0
  });

  useEffect(() => {
    // Connect to FastAPI WebSocket
    const ws = new WebSocket('ws://localhost:8000/ws/telemetry');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTelemetry(data);
    };

    ws.onclose = () => {
      setTelemetry((prev) => ({ ...prev, status: 'Backend Offline' }));
    };

    return () => ws.close();
  }, []);

  return (
    <div className="dashboard-container" style={{ display: 'flex', gap: '2rem', padding: '2rem', fontFamily: 'sans-serif', backgroundColor: '#1e1e1e', color: 'white', minHeight: '100vh' }}>
      
      {/* Video Feed Section */}
      <div className="video-section" style={{ flex: 2 }}>
        <h2>CoDrone EDU Vision Feed</h2>
        <div style={{ border: '2px solid #333', borderRadius: '8px', overflow: 'hidden' }}>
          <img 
            src="http://localhost:8000/video_feed" 
            alt="Live Camera Feed" 
            style={{ width: '100%', height: 'auto', display: 'block' }}
          />
        </div>
      </div>

      {/* Telemetry Panel */}
      <div className="telemetry-section" style={{ flex: 1, backgroundColor: '#2a2a2a', padding: '1.5rem', borderRadius: '8px' }}>
        <h2>Live Telemetry</h2>
        
        <div style={{ marginBottom: '1.5rem' }}>
          <h3 style={{ color: telemetry.status === 'Navigating' ? '#4caf50' : '#ff9800' }}>
            Status: {telemetry.status}
          </h3>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <p><strong>PID Roll Power:</strong> {telemetry.roll}</p>
          <progress value={telemetry.roll + 100} max="200" style={{ width: '100%' }}></progress>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888' }}>
            <span>-100 (Left)</span>
            <span>0</span>
            <span>+100 (Right)</span>
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <p><strong>PID Throttle Power:</strong> {telemetry.throttle}</p>
          <progress value={telemetry.throttle + 100} max="200" style={{ width: '100%' }}></progress>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888' }}>
            <span>-100 (Down)</span>
            <span>0</span>
            <span>+100 (Up)</span>
          </div>
        </div>
        
      </div>
    </div>
  );
}

export default App;