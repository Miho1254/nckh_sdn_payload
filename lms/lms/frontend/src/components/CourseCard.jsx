function CourseCard({ course, registered, onRegister }) {
    const isFull = course.available_slots <= 0;
    const fillPercent = Math.round((course.current_slots / course.max_slots) * 100);

    return (
        <div className="bg-slate-800/50 rounded-xl border border-slate-700/40 p-5 hover:border-slate-600/60 transition-all group">
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
                <div>
                    <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
                        {course.code}
                    </span>
                    <h3 className="text-white font-medium mt-2 leading-snug">{course.name}</h3>
                </div>
            </div>

            {/* Slot bar */}
            <div className="mb-4">
                <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-slate-400">Đã đăng ký</span>
                    <span className={isFull ? 'text-red-400' : 'text-slate-300'}>
                        {course.current_slots}/{course.max_slots}
                    </span>
                </div>
                <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                        className={`h-full rounded-full transition-all duration-500 ${isFull
                                ? 'bg-red-500'
                                : fillPercent > 80
                                    ? 'bg-amber-500'
                                    : 'bg-emerald-500'
                            }`}
                        style={{ width: `${fillPercent}%` }}
                    />
                </div>
            </div>

            {/* Action */}
            {registered ? (
                <div className="text-center py-2 rounded-lg bg-emerald-500/10 text-emerald-400 text-sm border border-emerald-500/20">
                    Đã đăng ký
                </div>
            ) : (
                <button
                    onClick={() => onRegister(course.id)}
                    disabled={isFull}
                    className={`w-full py-2.5 rounded-lg text-sm font-medium transition-all
            ${isFull
                            ? 'bg-slate-700/50 text-slate-500 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/10 hover:shadow-blue-500/20'
                        }`}
                >
                    {isFull ? 'Hết chỗ' : 'Đăng ký'}
                </button>
            )}
        </div>
    );
}

export default CourseCard;
