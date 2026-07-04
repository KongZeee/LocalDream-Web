import { NavLink, useLocation } from 'react-router-dom';
import {
  Box, Image, History, Zap, Settings, ChevronLeft, ChevronRight,
} from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import StatusBar from '@/components/StatusBar';

const navItems = [
  { to: '/', icon: Box, label: '模型管理' },
  { to: '/generate', icon: Image, label: '图像生成' },
  { to: '/history', icon: History, label: '历史记录' },
  { to: '/upscale', icon: Zap, label: '超分辨率' },
  { to: '/settings', icon: Settings, label: 'API 设置' },
];

export default function Sidebar() {
  const location = useLocation();
  const { sidebarCollapsed, toggleSidebar } = useAppStore();

  return (
    <aside
      className={`h-full flex flex-col bg-surface-light border-r border-surface-border transition-all duration-300 ${
        sidebarCollapsed ? 'w-16' : 'w-56'
      }`}
    >
      <div className="flex items-center h-16 px-4 border-b border-surface-border">
        {!sidebarCollapsed && (
          <span className="font-display text-lg font-bold text-neon-purple tracking-wider">
            LOCAL DREAM
          </span>
        )}
        <button
          onClick={toggleSidebar}
          className={`text-gray-500 hover:text-neon-purple transition-colors ${
            sidebarCollapsed ? 'mx-auto' : 'ml-auto'
          }`}
        >
          {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      <nav className="flex-1 py-4">
        {navItems.map((item) => {
          const isActive = location.pathname === item.to ||
            (item.to !== '/' && location.pathname.startsWith(item.to));
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={`flex items-center gap-3 px-4 py-3 mx-2 my-1 rounded-lg transition-all duration-200 group ${
                isActive
                  ? 'bg-neon-purple/10 text-neon-purple shadow-glow-purple'
                  : 'text-gray-500 hover:text-neon-purple hover:bg-surface-lighter'
              }`}
            >
              <item.icon size={20} />
              {!sidebarCollapsed && (
                <span className="font-mono text-sm">{item.label}</span>
              )}
            </NavLink>
          );
        })}
      </nav>

      <div className="p-4 border-t border-surface-border space-y-3">
        <StatusBar />
        {!sidebarCollapsed && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-neon-cyan animate-pulse" />
            <span className="text-xs text-gray-500 font-mono">在线</span>
          </div>
        )}
      </div>
    </aside>
  );
}