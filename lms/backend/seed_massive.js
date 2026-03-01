const { pool, initDB } = require('./db');
const bcrypt = require('bcryptjs');
const { faker } = require('@faker-js/faker');

const STUDENT_COUNT = 5000;
const BATCH_SIZE = 500; // Chèn mỗi lô 500 records để tránh sập Memory

const COURSES = [
    { code: 'CS101', name: 'Nhập môn Lập trình', max_slots: 100 },
    { code: 'CS201', name: 'Cấu trúc Dữ liệu & Giải thuật', max_slots: 80 },
    { code: 'CS301', name: 'Mạng máy tính', max_slots: 70 },
    { code: 'CS302', name: 'Hệ điều hành', max_slots: 60 },
    { code: 'CS401', name: 'Trí tuệ Nhân tạo', max_slots: 50 },
    { code: 'CS402', name: 'Machine Learning', max_slots: 50 },
    { code: 'CS403', name: 'An toàn Thông tin', max_slots: 50 },
    { code: 'MATH101', name: 'Giải tích 1', max_slots: 120 },
    { code: 'MATH201', name: 'Đại số Tuyến tính', max_slots: 100 },
    { code: 'MATH301', name: 'Xác suất & Thống kê', max_slots: 90 },
    { code: 'PHY101', name: 'Vật lý Đại cương 1', max_slots: 100 },
    { code: 'PHY201', name: 'Vật lý Đại cương 2', max_slots: 90 },
    { code: 'ENG101', name: 'Tiếng Anh Cơ bản', max_slots: 150 },
    { code: 'ENG201', name: 'Tiếng Anh Chuyên ngành', max_slots: 100 },
    { code: 'CS501', name: 'Đồ án Chuyên ngành', max_slots: 30 },
    { code: 'CS502', name: 'Seminar Khoa học', max_slots: 40 },
    { code: 'CS303', name: 'Cơ sở Dữ liệu', max_slots: 80 },
    { code: 'CS304', name: 'Lập trình Web', max_slots: 80 },
    { code: 'CS305', name: 'Phát triển Ứng dụng Di động', max_slots: 70 },
    { code: 'CS306', name: 'Cloud Computing', max_slots: 60 },
];

async function seedMassive() {
    await initDB();
    const client = await pool.connect();

    try {
        console.log("🔥 Bắt đầu quy trình Seeding Massive...");
        await client.query('BEGIN');

        // 1. Wipe dữ liệu cũ
        console.log("Xóa dữ liệu cũ...");
        await client.query('DELETE FROM registrations');
        await client.query('DELETE FROM students');
        await client.query('DELETE FROM courses');

        await client.query('ALTER SEQUENCE students_id_seq RESTART WITH 1');
        await client.query('ALTER SEQUENCE courses_id_seq RESTART WITH 1');
        await client.query('ALTER SEQUENCE registrations_id_seq RESTART WITH 1');

        // 2. Seed Môn học (Khống chế max_slots tạo Row Locking)
        console.log("Đang tạo Môn học...");
        for (const c of COURSES) {
            await client.query(
                'INSERT INTO courses (code, name, max_slots) VALUES ($1, $2, $3)',
                [c.code, c.name, c.max_slots]
            );
        }
        console.log(`✅ Đã tạo ${COURSES.length} môn học.`);

        // 3. Seed 5000 Sinh viên (Batch Insert)
        console.log(`Đang đúc ${STUDENT_COUNT} Sinh viên ảo (Cẩn thận I/O!)...`);
        const hash = bcrypt.hashSync('password123', 4); // Hash sẵn 1 pass chung cho tất cả để tiết kiệm time

        let batchValues = [];
        let batchQueryIndexes = [];
        let queryParamsCount = 1;

        for (let i = 1; i <= STUDENT_COUNT; i++) {
            // Dùng Faker.js tạo data thực tế
            const username = `sv${String(i).padStart(6, '0')}`; // Đảm bảo ID tuần tự từ sv000001 -> sv005000 để dễ bào load test
            const fullName = faker.person.fullName();

            // Ý tưởng "Góp ý": Thêm trường Priority/Credits (Có thể alter table jsonb hoặc dùng tạm để sinh behavior)
            // Ở đây ta mô phỏng logic chèn trước, cấu trúc CSDL hiện tại chỉ yêu cầu username/name/password.

            batchValues.push(username, fullName, hash);
            batchQueryIndexes.push(`($${queryParamsCount}, $${queryParamsCount + 1}, $${queryParamsCount + 2})`);
            queryParamsCount += 3;

            // Xả Batch khi đủ 500 dòng
            if (i % BATCH_SIZE === 0 || i === STUDENT_COUNT) {
                const queryText = `INSERT INTO students (username, full_name, password) VALUES ${batchQueryIndexes.join(', ')}`;
                await client.query(queryText, batchValues);

                console.log(`   -> Đã chèn thành công ${i}/${STUDENT_COUNT} sinh viên...`);

                // Reset batch
                batchValues = [];
                batchQueryIndexes = [];
                queryParamsCount = 1;
            }
        }
        console.log(`✅ Đã tạo xong ${STUDENT_COUNT} sinh viên ảo!`);

        await client.query('COMMIT');
        console.log('🎉 QUÁ TRÌNH SEEDING HOÀN TẤT THÀNH CÔNG!');
    } catch (err) {
        await client.query('ROLLBACK');
        console.error('❌ LỖI SEEDING:', err.message);
    } finally {
        client.release();
        await pool.end();
    }
}

seedMassive();
