import { useEffect, useState, useRef } from 'react';
import './App.css';

function App() {
  const [telemetry, setTelemetry] = useState({
    status: 'Disconnected',
    roll: 0, throttle: 0, pitch: 0, yaw: 0, battery: 0,
    shield_active: false
  });
  
  const wsRef = useRef<WebSocket | null>(null);
  
  // Track currently held keys for smooth diagonal movement
  const activeKeys = useRef(new Set());

  useEffect(() => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws/telemetry');

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTelemetry(data);
    };

    wsRef.current.onclose = () => {
      setTelemetry((prev) => ({ ...prev, status: 'Backend Offline' }));
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Global Keyboard Listener for WASD Steering and Emergency Stop
  useEffect(() => {
    const sendSteering = () => {
      let pitch = 0;
      let yaw = 0;
      
      const power = 5; // Speed for keyboard commands (0 to 100)

      if (activeKeys.current.has('KeyW')) pitch += power;
      if (activeKeys.current.has('KeyS')) pitch -= power;
      if (activeKeys.current.has('KeyA')) yaw -= power;
      if (activeKeys.current.has('KeyD')) yaw += power;
      

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command: 'steer', pitch, yaw }));
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
    // ... (keep Takeoff, Land, Flip, and Spacebar checks)

    let newPitch = telemetry.pitch;
    let newYaw = telemetry.yaw;
    let newThrottle = telemetry.throttle;
    let newRoll = telemetry.roll;

    // Standard WASD steering
    if (e.code === 'KeyW') newPitch = 40;  
    if (e.code === 'KeyS') newPitch = -40; 
    if (e.code === 'KeyA') newYaw = -50;   
    if (e.code === 'KeyD') newYaw = 50;    
    
    // Altitude Controls
    if (e.code === 'KeyE') newThrottle = 40;        // Ascend
    if (e.code === 'ShiftLeft') newThrottle = -40;  // Descend (Go Down)

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ 
          command: 'steer', pitch: newPitch, yaw: newYaw, roll: newRoll, throttle: newThrottle 
      }));
    }
  };

  const handleKeyUp = (e: KeyboardEvent) => {
    let newPitch = telemetry.pitch;
    let newYaw = telemetry.yaw;
    let newThrottle = telemetry.throttle;
    let newRoll = telemetry.roll;

    if (e.code === 'KeyW' || e.code === 'KeyS') newPitch = 0;
    if (e.code === 'KeyA' || e.code === 'KeyD') newYaw = 0;
    if (e.code === 'KeyE' || e.code === 'ShiftLeft') newThrottle = 0;

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ 
          command: 'steer', pitch: newPitch, yaw: newYaw, roll: newRoll, throttle: newThrottle 
      }));
    }
  };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  return (
    <div className="dashboard-container" style={{ display: 'flex', gap: '2rem', padding: '2rem', fontFamily: 'sans-serif', backgroundColor: '#1e1e1e', color: 'white', minHeight: '100vh' }}>
      
      {/* Video Feed Section */}
      <div className="video-section" style={{ flex: 2 }}>
        <h2>CoDrone EDU Vision Feed</h2>
        <div style={{ border: '2px solid #333', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
          
          {telemetry.status.includes('EMERGENCY') && (
            <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(255,0,0,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '3rem', fontWeight: 'bold' }}>
              MOTORS KILLED
            </div>
          )}

          {telemetry.shield_active && !telemetry.status.includes('EMERGENCY') && (
            <div style={{ position: 'absolute', top: '20px', left: '50%', transform: 'translateX(-50%)', backgroundColor: 'rgba(255, 152, 0, 0.8)', padding: '10px 20px', borderRadius: '8px', color: 'white', fontSize: '1.5rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '10px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)' }}>
              SHIELD ACTIVE
            </div>
          )}
          
          <img 
            src="http://localhost:8000/video_feed" 
            alt="Live Camera Feed" 
            style={{ width: '100%', height: 'auto', display: 'block' }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem', fontWeight: 'bold' }}>
          <p style={{ color: '#4caf50' }}>[W, A, S, D] TO STEER (PITCH & YAW)</p>
          <p style={{ color: '#ff5252' }}>⚠️ PRESS [SPACEBAR] TO EMERGENCY STOP</p>
        </div>
      </div>

      {/* Telemetry Panel */}
      <div className="telemetry-section" style={{ flex: 1, backgroundColor: '#2a2a2a', padding: '1.5rem', borderRadius: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2>Live Telemetry</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
             <span style={{ fontSize: '1.2rem', color: telemetry.battery > 20 ? '#4caf50' : '#ff5252' }}>
               🔋 {telemetry.battery}%
             </span>
          </div>
        </div>
        
        <div style={{ marginBottom: '1.5rem' }}>
          <h3 style={{ 
            color: 
              telemetry.status === 'NAVIGATING' ? '#4caf50' : // Green
              telemetry.status === 'HOVERING' ? '#2196f3' :   // Blue
              telemetry.status === 'TAKING_OFF' ? '#ff9800' : // Orange
              telemetry.status === 'EMERGENCY' ? '#ff5252' :  // Red
              '#888888' // Gray for LANDED
          }}>
            State: {telemetry.status}
          </h3>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <p><strong>PID Roll Power (Camera X-Axis):</strong> {telemetry.roll}</p>
          <progress value={telemetry.roll + 100} max="200" style={{ width: '100%' }}></progress>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888' }}>
            <span>-100 (Left)</span><span>0</span><span>+100 (Right)</span>
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <p><strong>PID Throttle Power (Camera Y-Axis):</strong> {telemetry.throttle}</p>
          <progress value={telemetry.throttle + 100} max="200" style={{ width: '100%' }}></progress>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888' }}>
            <span>-100 (Down)</span><span>0</span><span>+100 (Up)</span>
          </div>
        </div>

        <hr style={{ borderColor: '#444', margin: '2rem 0' }}/>

        <div style={{ marginBottom: '1rem' }}>
          <p><strong>Keyboard Pitch Power (W/S):</strong> {telemetry.pitch}</p>
          <progress value={telemetry.pitch + 100} max="200" style={{ width: '100%' }}></progress>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888' }}>
            <span>-100 (Backwards)</span><span>0</span><span>+100 (Forwards)</span>
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <p><strong>Keyboard Yaw Power (A/D):</strong> {telemetry.yaw}</p>
          <progress value={telemetry.yaw + 100} max="200" style={{ width: '100%' }}></progress>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888' }}>
            <span>-100 (Rotate Left)</span><span>0</span><span>+100 (Rotate Right)</span>
          </div>
        </div>
        
      </div>
    </div>
  );
}

export default App;