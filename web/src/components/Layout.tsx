import { Outlet } from 'react-router-dom';
import Sidebar from '@/components/Sidebar';

export default function Layout() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface">
      <Sidebar />
      <main className="flex-1 overflow-auto relative">
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-neon-purple/20 to-transparent" />
        </div>
        <div className="p-6 animate-fade-in">
          <Outlet />
        </div>
      </main>
    </div>
  );
}