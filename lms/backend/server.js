const express = require('express');
const cors = require('cors');
const path = require('path');
const { initDB } = require('./db');

const authRoutes = require('./routes/auth');
const courseRoutes = require('./routes/courses');
const registerRoutes = require('./routes/register');
const authMiddleware = require('./middleware/auth');

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(cors());
app.use(express.json());

// Request logging (for monitoring during stress test)
app.use((req, _res, next) => {
    const ts = new Date().toISOString();
    console.log(`[${ts}] ${req.method} ${req.url}`);
    next();
});

// Serve frontend static files
app.use(express.static(path.join(__dirname, '../frontend/dist')));

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/courses', courseRoutes);
app.use('/api/register', authMiddleware, registerRoutes);

// Health check
app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', uptime: process.uptime() });
});

// SPA fallback
app.get('*', (_req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/dist/index.html'));
});

// Error handler
app.use((err, _req, res, _next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server after DB init
initDB()
    .then(() => {
        app.listen(PORT, '0.0.0.0', () => {
            console.log(`LMS Backend running on http://0.0.0.0:${PORT}`);
            console.log(`Database: ${process.env.DB_HOST || '10.0.0.6'}:5432/lms`);
        });
    })
    .catch((err) => {
        console.error('Failed to initialize DB:', err.message);
        process.exit(1);
    });
