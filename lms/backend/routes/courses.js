const express = require('express');
const { pool } = require('../db');

const router = express.Router();

// GET /api/courses — Danh sách tất cả môn học + slots còn trống
router.get('/', async (_req, res) => {
  try {
    const { rows } = await pool.query(`
      SELECT id, code, name, max_slots, current_slots,
             (max_slots - current_slots) AS available_slots
      FROM courses
      ORDER BY code
    `);
    res.json(rows);
  } catch (err) {
    console.error('Courses error:', err.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/courses/:id — Chi tiết 1 môn
router.get('/:id', async (req, res) => {
  try {
    const { rows } = await pool.query(`
      SELECT id, code, name, max_slots, current_slots,
             (max_slots - current_slots) AS available_slots
      FROM courses
      WHERE id = $1
    `, [req.params.id]);

    if (rows.length === 0) {
      return res.status(404).json({ error: 'Course not found' });
    }

    res.json(rows[0]);
  } catch (err) {
    console.error('Course detail error:', err.message);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
