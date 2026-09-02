const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

app.use(express.static('public'));

let prevVolume = 0;

io.on('connection', (socket) => {
  socket.on('sensor-data', (data) => {
    const { expression, volume, pitchVar } = data;

    // --- 1. 音声ストレス算出 ---
    // 声量の急激な変化（声のひっくり返り）とピッチのゆらぎを計算
    const volChange = Math.abs(volume - prevVolume);
    prevVolume = volume;
    const voiceStress = Math.min(100, (volChange * 1.5) + (pitchVar * 0.8));

    // --- 2. 表情ストレス算出 ---
    let faceStress = 10; // 真顔
    if (expression === "Angry") faceStress = 85;      // 焦り・怒り
    else if (expression === "Smile") faceStress = 45; // 不自然な緊張・笑顔

    // --- 3. 統合アルゴリズム ＆ ポリグラフ用ノイズ算出 ---
    // 音声 50% + 表情 50% に、ポリグラフの針の揺れ（±4点）を加算
    const rawScore = (voiceStress * 0.5) + (faceStress * 0.5);
    const jitter = (Math.random() - 0.5) * 8;
    const finalScore = Math.min(100, Math.max(0, Math.round(rawScore + jitter)));

    // 判定ステータス
    let status = "😐 正常（落ち着いている）";
    if (finalScore > 70) status = "🚨 CRITICAL（動揺・嘘の兆候）";
    else if (finalScore > 40) status = "⚠️ SUSPICIOUS（やや不自然）";

    // ブラウザへリアルタイム返送
    socket.emit('lie-score-update', {
      score: finalScore,
      status: status
    });
  });
});

server.listen(3000, () => {
  console.log("嘘発見器サーバー起動: http://localhost:3000");
});