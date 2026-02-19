const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { pool } = require('../db');

const router = express.Router();
const JWT_SECRET = process.env.JWT_SECRET || 'nckh-sdn-secret-key';

// POST /api/auth/login
router.post('/login', async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ error: 'Username and password required' });
    }

    try {
        const { rows } = await pool.query(
            'SELECT * FROM students WHERE username = $1',
            [username]
        );
        const student = rows[0];

        if (!student) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const valid = bcrypt.compareSync(password, student.password);
        if (!valid) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const token = jwt.sign(
            { id: student.id, username: student.username },
            JWT_SECRET,
            { expiresIn: '2h' }
        );

        res.json({
            token,
            student: {
                id: student.id,
                username: student.username,
                full_name: student.full_name,
            },
        });
    } catch (err) {
        console.error('Login error:', err.message);
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = router;
module.exports.JWT_SECRET = JWT_SECRET;
