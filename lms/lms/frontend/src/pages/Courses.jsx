import { useState, useEffect } from 'react';
import CourseCard from '../components/CourseCard';

function Courses({ token }) {
    const [courses, setCourses] = useState([]);
    const [myRegs, setMyRegs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [toast, setToast] = useState(null);

    const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
    };

    const showToast = (msg, type = 'success') => {
        setToast({ msg, type });
        setTimeout(() => setToast(null), 3000);
    };

    const fetchCourses = async () => {
        try {
            const res = await fetch('/api/courses');
            const data = await res.json();
            setCourses(data);
        } catch {
            showToast('Lỗi tải danh sách môn học', 'error');
        }
    };

    const fetchMyRegs = async () => {
        try {
            const res = await fetch('/api/register/my', { headers });
            const data = await res.json();
            setMyRegs(data);
        } catch {
            // silent
        }
    };

    useEffect(() => {
        Promise.all([fetchCourses(), fetchMyRegs()]).then(() => setLoading(false));
        // Poll courses every 5s during stress test
        const interval = setInterval(fetchCourses, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleRegister = async (courseId) => {
        try {
            const res = await fetch('/api/register', {
                method: 'POST',
                headers,
                body: JSON.stringify({ course_id: courseId }),
            });
            const data = await res.json();

            if (!res.ok) {
                showToast(data.error, 'error');
                return;
            }

            showToast(`Đã đăng ký ${data.course_code} — còn ${data.slots_remaining} chỗ`);
            fetchCourses();
            fetchMyRegs();
        } catch {
            showToast('Lỗi kết nối server', 'error');
        }
    };

    const handleCancel = async (regId) => {
        try {
            const res = await fetch(`/api/register/${regId}`, {
                method: 'DELETE',
                headers,
            });

            if (!res.ok) {
                const data = await res.json();
                showToast(data.error, 'error');
                return;
            }

            showToast('Đã hủy đăng ký');
            fetchCourses();
            fetchMyRegs();
        } catch {
            showToast('Lỗi kết nối server', 'error');
        }
    };

    const registeredIds = new Set(myRegs.map((r) => r.code));

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            </div>
        );
    }

    return (
        <div>
            {/* Toast */}
            {toast && (
                <div
                    className={`fixed top-4 right-4 z-50 px-5 py-3 rounded-xl text-sm font-medium shadow-xl transition-all animate-slide-in
            ${toast.type === 'error'
                            ? 'bg-red-500/90 text-white'
                            : 'bg-emerald-500/90 text-white'
                        }`}
                >
                    {toast.msg}
                </div>
            )}

            {/* My Registrations */}
            {myRegs.length > 0 && (
                <section className="mb-8">
                    <h2 className="text-lg font-semibold text-white mb-4">
                        Môn đã đăng ký ({myRegs.length})
                    </h2>
                    <div className="bg-slate-800/40 rounded-xl border border-slate-700/50 overflow-hidden">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-slate-400 border-b border-slate-700/50">
                                    <th className="text-left px-4 py-3 font-medium">Mã</th>
                                    <th className="text-left px-4 py-3 font-medium">Tên môn</th>
                                    <th className="text-left px-4 py-3 font-medium">Thời gian</th>
                                    <th className="px-4 py-3"></th>
                                </tr>
                            </thead>
                            <tbody>
                                {myRegs.map((r) => (
                                    <tr key={r.id} className="border-b border-slate-700/30 hover:bg-slate-700/20 transition-colors">
                                        <td className="px-4 py-3 font-mono text-blue-400">{r.code}</td>
                                        <td className="px-4 py-3 text-slate-200">{r.name}</td>
                                        <td className="px-4 py-3 text-slate-500">{r.registered_at}</td>
                                        <td className="px-4 py-3 text-right">
                                            <button
                                                onClick={() => handleCancel(r.id)}
                                                className="px-3 py-1 rounded-lg text-xs bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20 transition-colors"
                                            >
                                                Hủy
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>
            )}

            {/* Course List */}
            <section>
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-white">
                        Danh sách môn học ({courses.length})
                    </h2>
                    <button
                        onClick={fetchCourses}
                        className="text-xs px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
                    >
                        Refresh
                    </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {courses.map((course) => (
                        <CourseCard
                            key={course.id}
                            course={course}
                            registered={registeredIds.has(course.code)}
                            onRegister={handleRegister}
                        />
                    ))}
                </div>
            </section>
        </div>
    );
}

export default Courses;
