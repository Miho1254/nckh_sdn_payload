const express = require('express');
const { pool } = require('../db');

const router = express.Router();

// ── POST /api/register — STRESS POINT ──────────────────────
// Transaction: check slot → check duplicate → insert → update slot
// PostgreSQL handles row-level locks via FOR UPDATE

router.post('/', async (req, res) => {
    const studentId = req.user.id;
    const { course_id } = req.body;

    if (!course_id) {
        return res.status(400).json({ error: 'course_id is required' });
    }

    const client = await pool.connect();
    try {
        await client.query('BEGIN');

        // 1. Kiểm tra môn học + lock row (FOR UPDATE ngăn race condition)
        const { rows: courseRows } = await client.query(
            'SELECT id, code, name, max_slots, current_slots FROM courses WHERE id = $1 FOR UPDATE',
            [course_id]
        );
        const course = courseRows[0];

        if (!course) {
            await client.query('ROLLBACK');
            return res.status(404).json({ error: 'Course not found' });
        }

        if (course.current_slots >= course.max_slots) {
            await client.query('ROLLBACK');
            return res.status(409).json({
                error: `${course.code} is full (${course.max_slots}/${course.max_slots})`,
            });
        }

        // 2. Kiểm tra trùng
        const { rows: dupRows } = await client.query(
            'SELECT id FROM registrations WHERE student_id = $1 AND course_id = $2',
            [studentId, course_id]
        );
        if (dupRows.length > 0) {
            await client.query('ROLLBACK');
            return res.status(409).json({ error: 'Already registered for this course' });
        }

        // 3. Insert + update slot
        const { rows: regRows } = await client.query(
            'INSERT INTO registrations (student_id, course_id) VALUES ($1, $2) RETURNING id',
            [studentId, course_id]
        );
        await client.query(
            'UPDATE courses SET current_slots = current_slots + 1 WHERE id = $1',
            [course_id]
        );

        await client.query('COMMIT');

        res.status(201).json({
            registration_id: regRows[0].id,
            course_code: course.code,
            course_name: course.name,
            slots_remaining: course.max_slots - course.current_slots - 1,
        });
    } catch (err) {
        await client.query('ROLLBACK');
        if (err.code === '23505') {
            return res.status(409).json({ error: 'Already registered' });
        }
        console.error('Registration error:', err.message);
        res.status(500).json({ error: 'Internal server error' });
    } finally {
        client.release();
    }
});

// ── DELETE /api/register/:id — Hủy đăng ký ────────────────

router.delete('/:id', async (req, res) => {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');

        const { rows } = await client.query(
            'SELECT course_id FROM registrations WHERE id = $1 AND student_id = $2',
            [req.params.id, req.user.id]
        );

        if (rows.length === 0) {
            await client.query('ROLLBACK');
            return res.status(404).json({ error: 'Registration not found' });
        }

        await client.query(
            'DELETE FROM registrations WHERE id = $1 AND student_id = $2',
            [req.params.id, req.user.id]
        );
        await client.query(
            'UPDATE courses SET current_slots = current_slots - 1 WHERE id = $1 AND current_slots > 0',
            [rows[0].course_id]
        );

        await client.query('COMMIT');
        res.json({ message: 'Registration cancelled' });
    } catch (err) {
        await client.query('ROLLBACK');
        console.error('Cancel error:', err.message);
        res.status(500).json({ error: 'Internal server error' });
    } finally {
        client.release();
    }
});

// ── GET /api/register/my — Xem môn đã đăng ký ─────────────

router.get('/my', async (req, res) => {
    try {
        const { rows } = await pool.query(`
      SELECT r.id, r.registered_at, c.code, c.name
      FROM registrations r
      JOIN courses c ON r.course_id = c.id
      WHERE r.student_id = $1
      ORDER BY r.registered_at DESC
    `, [req.user.id]);

        res.json(rows);
    } catch (err) {
        console.error('My registrations error:', err.message);
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = router;
