const { pool, initDB } = require('./db');
const bcrypt = require('bcryptjs');

const STUDENT_COUNT = 500;
const COURSES = [
    { code: 'CS101', name: 'Nhập môn Lập trình', max_slots: 40 },
    { code: 'CS201', name: 'Cấu trúc Dữ liệu & Giải thuật', max_slots: 35 },
    { code: 'CS301', name: 'Mạng máy tính', max_slots: 30 },
    { code: 'CS302', name: 'Hệ điều hành', max_slots: 35 },
    { code: 'CS401', name: 'Trí tuệ Nhân tạo', max_slots: 30 },
    { code: 'CS402', name: 'Machine Learning', max_slots: 25 },
    { code: 'CS403', name: 'An toàn Thông tin', max_slots: 35 },
    { code: 'MATH101', name: 'Giải tích 1', max_slots: 50 },
    { code: 'MATH201', name: 'Đại số Tuyến tính', max_slots: 50 },
    { code: 'MATH301', name: 'Xác suất & Thống kê', max_slots: 45 },
    { code: 'PHY101', name: 'Vật lý Đại cương 1', max_slots: 45 },
    { code: 'PHY201', name: 'Vật lý Đại cương 2', max_slots: 40 },
    { code: 'ENG101', name: 'Tiếng Anh Cơ bản', max_slots: 50 },
    { code: 'ENG201', name: 'Tiếng Anh Chuyên ngành', max_slots: 40 },
    { code: 'CS501', name: 'Đồ án Chuyên ngành', max_slots: 20 },
    { code: 'CS502', name: 'Seminar Khoa học', max_slots: 30 },
    { code: 'CS303', name: 'Cơ sở Dữ liệu', max_slots: 35 },
    { code: 'CS304', name: 'Lập trình Web', max_slots: 40 },
    { code: 'CS305', name: 'Phát triển Ứng dụng Di động', max_slots: 30 },
    { code: 'CS306', name: 'Cloud Computing', max_slots: 25 },
];

async function seed() {
    await initDB();
    const client = await pool.connect();

    try {
        await client.query('BEGIN');

        // Clear
        await client.query('DELETE FROM registrations');
        await client.query('DELETE FROM students');
        await client.query('DELETE FROM courses');

        // Reset sequences
        await client.query('ALTER SEQUENCE students_id_seq RESTART WITH 1');
        await client.query('ALTER SEQUENCE courses_id_seq RESTART WITH 1');
        await client.query('ALTER SEQUENCE registrations_id_seq RESTART WITH 1');

        // Seed courses
        for (const c of COURSES) {
            await client.query(
                'INSERT INTO courses (code, name, max_slots) VALUES ($1, $2, $3)',
                [c.code, c.name, c.max_slots]
            );
        }
        console.log(`  ${COURSES.length} courses created`);

        // Seed students (bcrypt cost=4 for speed)
        const hash = bcrypt.hashSync('password123', 4);
        for (let i = 1; i <= STUDENT_COUNT; i++) {
            const username = `sv${String(i).padStart(4, '0')}`;
            const fullName = `Sinh Vien ${i}`;
            await client.query(
                'INSERT INTO students (username, full_name, password) VALUES ($1, $2, $3)',
                [username, fullName, hash]
            );
            if (i % 100 === 0) console.log(`  ${i}/${STUDENT_COUNT} students...`);
        }
        console.log(`  ${STUDENT_COUNT} students created (password: password123)`);

        await client.query('COMMIT');
        console.log('Seed complete!');
    } catch (err) {
        await client.query('ROLLBACK');
        console.error('Seed failed:', err.message);
    } finally {
        client.release();
        await pool.end();
    }
}

seed();
