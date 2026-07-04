import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from '@/components/Layout';
import ModelListPage from '@/pages/ModelListPage';
import GeneratePage from '@/pages/GeneratePage';
import HistoryPage from '@/pages/HistoryPage';
import UpscalePage from '@/pages/UpscalePage';
import SettingsPage from '@/pages/SettingsPage';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<ModelListPage />} />
          <Route path="/generate" element={<GeneratePage />} />
          <Route path="/generate/:modelId" element={<GeneratePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/upscale" element={<UpscalePage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </Router>
  );
}