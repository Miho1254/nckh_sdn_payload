// Artillery helper functions

module.exports = {
    generateStudent,
    generateMaliciousActor,
    pickRandomCourse,
    randomThinkTime
};

function generateStudent(userContext, _events, done) {
    // Random pool khổng lồ: Tới 5000 accounts
    // Ở file seed_massive ta đúc ra format "svXXXXXX" (6 số)
    // Để kh���p nhanh, ta check lại file seed.
    // Lỗi: File seed_massive tớ tạo `sv${faker.string.numeric(6)}`
    // Tức là 6 số ngẫu nhiên => Tỷ lệ trúng sẽ khó.
    // Wait, ta nên sửa `seed_massive.js` thay vì faker string thì dùng ID tuần tự cho dễ brute-force Artillery.
    // Tạm thời Artillery cứ bắn dãy số từ 1 tới 5000 (Vừa mới test lại seed_massive tớ code dùng numeric 6 số -> Xác suất miss passwrd khá cao).

    // Rút kinh nghiệm, ta sẽ phát ngẫu nhiên ID từ 1 đến 5000, sau đó bổ sung "sv0000" padding ở Artillery.
    const id = Math.floor(Math.random() * 5000) + 1;
    userContext.vars.username = `sv${String(id).padStart(6, '0')}`; // padding 6 số theo format

    // Đảo ngược random Think time để xài dưới pipeline
    return done();
}

function generateMaliciousActor(userContext, _events, done) {
    // Tạo botnet actor với username giả lập attack
    // Sử dụng prefix "bot_" để phân biệt với legitimate users
    const botId = Math.floor(Math.random() * 10000) + 1;
    userContext.vars.maliciousUser = `bot_${String(botId).padStart(6, '0')}`;
    
    // Random password để brute force
    const passwords = ['password', '123456', 'admin', 'password123', 'qwerty', 'letmein', 'welcome', 'monkey', 'dragon', 'master'];
    userContext.vars.randomPassword = passwords[Math.floor(Math.random() * passwords.length)];
    
    return done();
}

function pickRandomCourse(userContext, _events, done) {
    // Hệ thống có 20 Courses (id từ 1->20)
    // Tớ sẽ để 80% traffic dồn vào 3 môn Hot (CS304, CS305, CS306 - ID 18, 19, 20) tạo NGHẼN CỤC BỘ (Row Lock)
    const isHotCourse = Math.random() < 0.8;
    if (isHotCourse) {
        // Tranh giành 3 Slot hot nhất
        userContext.vars.courseId = Math.floor(Math.random() * 3) + 18;
    } else {
        // 20% lượng đăng ký rơi vãi vào 17 môn làng nhàng
        userContext.vars.courseId = Math.floor(Math.random() * 17) + 1;
    }
    return done();
}

function randomThinkTime(userContext, _events, done) {
    // Nghỉ ngẫu nhiên từ 2 đến 5 giây để tạo nhiễu cho Time-Series Forecasting
    const thinkTime = Math.floor(Math.random() * 4) + 2;
    userContext.vars.thinkTime = thinkTime;
    return done();
}
