import { Routes, Route } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import Home from './pages/Home';

const Meeting = lazy(() => import('./pages/Meeting'));

function App() {
  return (
    <div className="min-h-screen bg-dark-900 text-white">
      <Suspense fallback={null}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/meeting/:roomId" element={<Meeting />} />
        </Routes>
      </Suspense>
    </div>
  );
}

export default App;
