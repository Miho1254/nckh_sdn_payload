import { useState } from 'react';
import Login from './pages/Login';
import Courses from './pages/Courses';

function App() {
    const [token, setToken] = useState(null);
    const [user, setUser] = useState(null);

    const handleLogin = (tkn, studentData) => {
        setToken(tkn);
        setUser(studentData);
    };

    const handleLogout = () => {
        setToken(null);
        setUser(null);
    };

    if (!token) {
        return <Login onLogin={handleLogin} />;
    }

    return (
        <div className="min-h-screen">
            {/* Header */}
            <header className="bg-slate-800/80 backdrop-blur-sm border-b border-slate-700 sticky top-0 z-10">
                <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-white font-bold text-sm">
                            LMS
                        </div>
                        <h1 className="text-lg font-semibold text-white">Đăng ký Học phần</h1>
                    </div>
                    <div className="flex items-center gap-4">
                        <span className="text-sm text-slate-400">
                            {user?.full_name} <span className="text-slate-600">({user?.username})</span>
                        </span>
                        <button
                            onClick={handleLogout}
                            className="text-sm px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
                        >
                            Đăng xuất
                        </button>
                    </div>
                </div>
            </header>

            {/* Main */}
            <main className="max-w-6xl mx-auto px-6 py-8">
                <Courses token={token} />
            </main>
        </div>
    );
}

export default App;
