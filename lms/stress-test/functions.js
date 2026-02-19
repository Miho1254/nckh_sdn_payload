// Artillery helper functions

module.exports = {
    generateStudent,
    pickRandomCourse,
};

function generateStudent(userContext, _events, done) {
    const id = Math.floor(Math.random() * 500) + 1;
    userContext.vars.username = `sv${String(id).padStart(4, '0')}`;
    return done();
}

function pickRandomCourse(userContext, _events, done) {
    const courseId = Math.floor(Math.random() * 20) + 1;
    userContext.vars.courseId = courseId;
    return done();
}
