const { Pool } = require('pg');

const pool = new Pool({
  host: process.env.DB_HOST || '10.0.0.6',   // h6 trong Mininet
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'lms',
  user: process.env.DB_USER || 'lms',
  password: process.env.DB_PASS || 'lms123',
  max: 20,                                    // connection pool size
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
});

// Init schema
async function initDB() {
  const client = await pool.connect();
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS courses (
        id SERIAL PRIMARY KEY,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        max_slots INTEGER NOT NULL,
        current_slots INTEGER DEFAULT 0
      );

      CREATE TABLE IF NOT EXISTS registrations (
        id SERIAL PRIMARY KEY,
        student_id INTEGER NOT NULL REFERENCES students(id),
        course_id INTEGER NOT NULL REFERENCES courses(id),
        registered_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(student_id, course_id)
      );

      CREATE INDEX IF NOT EXISTS idx_reg_student ON registrations(student_id);
      CREATE INDEX IF NOT EXISTS idx_reg_course ON registrations(course_id);
    `);
    console.log('Database schema initialized');
  } finally {
    client.release();
  }
}

module.exports = { pool, initDB };
