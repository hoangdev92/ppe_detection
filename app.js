// server/app.js
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const path = require('path');
const { Readable } = require('stream'); // for streaming fetch Response.body
const { URL } = require('url'); 

// ==== ADD: DB + mail + state for one-time alerts ====
const sequelize = require('./models');
const Violation = require('./models/Violation');
const { sendViolationMail } = require('./services/notify');

// clientId -> Map(trackId -> { missingKey: string, lastSeen: number })
const alertedByClient = new Map();
const TRACK_TTL_MS = 10_000;

function getClientMap(clientId) {
  let m = alertedByClient.get(clientId);
  if (!m) { m = new Map(); alertedByClient.set(clientId, m); }
  return m;
}
function cleanupClientMap(m) {
  const now = Date.now();
  for (const [tid, s] of m.entries()) {
    if ((now - (s.lastSeen || 0)) > TRACK_TTL_MS) m.delete(tid);
  }
}
// ====================================================
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PYTHON_PATH = 'python3';
const PY_SCRIPT = path.join(__dirname, 'python', 'inference.py');

const py = spawn(PYTHON_PATH, [PY_SCRIPT], { stdio: ['pipe', 'pipe', 'inherit'] });
py.stdout.setEncoding('utf8');

let pyStdoutBuf = '';
py.stdout.on('data', (chunk) => {
  pyStdoutBuf += chunk;
  let idx;
  while ((idx = pyStdoutBuf.indexOf('\n')) >= 0) {
    const line = pyStdoutBuf.slice(0, idx).trim();
    pyStdoutBuf = pyStdoutBuf.slice(idx + 1);
    if (!line) continue;
    let obj = null;
    try { obj = JSON.parse(line); } catch(e) { console.error('Bad JSON from python:', e, line); continue; }
    // forward inference result to client by clientId
    const { clientId, boxes } = obj;
    for (const [ws, meta] of clients.entries()) {
      if (meta.clientId === clientId && ws.readyState === WebSocket.OPEN) {
        // ws.send(JSON.stringify({ type: 'inference', boxes }));
        // break;
        ws.send(JSON.stringify({ type: 'inference', boxes }));

        // === One-time alert per trackId and missing-set ===
        const people = boxes.filter(b => (b.name === 'person') || b.class === 3);
        const itemsBy = {
          helmet: boxes.filter(b => b.name === 'helmet'),
          vest:   boxes.filter(b => b.name === 'vest'),
          glove:  boxes.filter(b => b.name === 'glove'),
          boots:  boxes.filter(b => b.name === 'boots'),
        };
        const iou = (a,b) => {
          const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
          const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
          const inter = Math.max(0, x2-x1) * Math.max(0, y2-y1);
          const areaA = Math.max(0, a.x2-a.x1) * Math.max(0, a.y2-a.y1);
          const areaB = Math.max(0, b.x2-b.x1) * Math.max(0, b.y2-b.y1);
          const uni = areaA + areaB - inter;
          return uni > 0 ? inter/uni : 0;
        };

        const CLASS_REQUIRED = ['helmet','vest','glove','boots'];
        const m = getClientMap(clientId);
        cleanupClientMap(m);

        const newAlerts = [];
        for (const p of people) {
          const tid = p.id; // track id from YOLO tracking
          if (tid == null) continue;

          const missing = [];
          for (const name of CLASS_REQUIRED) {
            const arr = itemsBy[name] || [];
            const ok = arr.some(it => iou(p, it) >= 0.1);
            if (!ok) missing.push(name);
          }

          const state = m.get(tid) || {};
          state.lastSeen = Date.now();
          if (!missing.length) {
            state.missingKey = '';
            m.set(tid, state);
            continue;
          }

          const key = missing.slice().sort().join(',');
          if (state.missingKey !== key) {
            state.missingKey = key;
            m.set(tid, state);
            newAlerts.push({ trackId: tid, missingItems: missing });
          }
        }

        if (newAlerts.length) {
          const allMissing = Array.from(new Set(newAlerts.flatMap(a => a.missingItems)));
          ws.send(JSON.stringify({
            type: 'violation',
            clientId,
            missingItems: allMissing,
            tracks: newAlerts.map(a => a.trackId)
          }));

          // (async () => {
          //   try {
          //     await Violation.create({
          //       clientId,
          //       missingItems: JSON.stringify(allMissing),
          //       frameWidth: typeof meta?.width === 'number' ? meta.width : null,
          //       frameHeight: typeof meta?.height === 'number' ? meta.height : null
          //     });
          //   } catch (e) {
          //     console.warn('Save violation failed:', e.message);
          //   }
          // })();

          // sendViolationMail({ clientId, missingItems: allMissing }).catch(()=>{});
        }
        break;
      }
    }
    // mark python free and try to flush next frame
    pyBusy = false;
    flushPendingToPython();
  }
});

let clients = new Map(); // ws -> { clientId, pendingMeta, pendingBinary (Buffer) }
let nextClientId = 1;

// concurrency control for python stdin: ensure only one frame is written at a time
let pyBusy = false;

function flushPendingToPython() {
  if (pyBusy) return;
  // pick any client that has pendingBinary
  for (const [ws, meta] of clients.entries()) {
    if (meta.pendingBinary && meta.pendingMeta) {
      const buf = meta.pendingBinary;
      const m = meta.pendingMeta;
      // write metadata JSON line containing clientId and len
      const header = JSON.stringify({ clientId: m.clientId, len: buf.length }) + '\n';
      try {
        pyBusy = true;
        py.stdin.write(header);
        // write raw bytes (Buffer)
        py.stdin.write(buf);
      } catch (e) {
        console.error('Failed to write to python stdin', e);
        pyBusy = false;
      }
      // clear pendingBinary after writing
      meta.pendingBinary = null;
      meta.pendingMeta = null;
      return; // only one at a time
    }
  }
}

wss.on('connection', (ws) => {
  const clientId = String(nextClientId++);
  clients.set(ws, { clientId, pendingMeta: null, pendingBinary: null });
  ws.send(JSON.stringify({ type: 'hello', clientId }));

  // store last received meta (for next binary)
  ws.on('message', (message, isBinary) => {
    const client = clients.get(ws);
    if (!client) return;
    if (isBinary) {
      // message is ArrayBuffer or Buffer
      const buf = Buffer.from(message);
      // If there is no meta waiting, ignore (or could accept default)
      if (!client.pendingMeta) {
        // drop if no meta available
        console.warn('Received binary with no meta — dropping');
        return;
      }
      // store latest binary (overwrite older)
      client.pendingBinary = buf;
      // enqueue for python
      flushPendingToPython();
    } else {
      // text message (expect JSON meta)
      try {
        const msg = JSON.parse(message.toString());
        if (msg.type === 'frame_meta') {
          // store meta: include clientId (prefer server assigned) and dimensions/size
          client.pendingMeta = {
            clientId: client.clientId,
            width: msg.width,
            height: msg.height,
            size: msg.size
          };
          // Note: we do not send to python until binary arrives
        } else {
          // other message types - ignore or handle
        }
      } catch (e) {
        console.warn('Invalid JSON text message from client', e);
      }
    }
  });

  ws.on('close', () => {
    clients.delete(ws);
  });
});

function validateProxyUrl(raw) {
  let u;
  try { u = new URL(raw); } catch { return null; }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
  const host = u.hostname.toLowerCase();
  return u;
}

function guessContentTypeByPath(u) {
  const ext = path.extname(u.pathname || '').toLowerCase();
  if (ext === '.mp4') return 'video/mp4';
  if (ext === '.webm') return 'video/webm';
  if (ext === '.ogg' || ext === '.ogv') return 'video/ogg';
  if (ext === '.m3u8') return 'application/vnd.apple.mpegurl'; // HLS
  if (ext === '.mpd') return 'application/dash+xml'; // DASH
  return null;
}

app.use(express.static(path.join(__dirname, '..', 'frontend', 'dist')));
app.get('/health', (req, res) => res.json({ ok: true }));

app.get('/proxy', async (req, res) => {
  const rawUrl = req.query.url;
  const u = validateProxyUrl(rawUrl);
  if (!u) return res.status(400).send('Invalid or not whitelisted URL');

  // Forward a subset of headers (Range for seeking)
  const headers = {};
  if (req.headers.range) headers['Range'] = req.headers.range;
  if (req.headers['user-agent']) headers['User-Agent'] = req.headers['user-agent'];

  // Timeout để tránh treo kết nối
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), 25_000);


  let r;
  try {
    r = await fetch(u.toString(), { headers, redirect: 'follow', signal: ac.signal });
  } catch (e) {
    clearTimeout(t);
    return res.status(502).send('Upstream fetch failed');
  }
  clearTimeout(t);

  if (!r.ok && r.status !== 206) {
    // 206 là Partial Content cho Range
    return res.status(r.status).send('Upstream responded ' + r.status);
  }

  // Set CORS cho canvas/Video
  res.set('Access-Control-Allow-Origin', '*'); // hoặc set origin cụ thể nếu cần
  res.set('Cross-Origin-Resource-Policy', 'cross-origin');

  const upstreamCT = r.headers.get('content-type');
  const guessedCT = upstreamCT || guessContentTypeByPath(u) || 'application/octet-stream';

  // Forward các header video quan trọng
  res.set('Content-Type', guessedCT);

  const cr = r.headers.get('content-range');
  if (cr) res.set('Content-Range', cr);

  const cl = r.headers.get('content-length');
  if (cl) res.set('Content-Length', cl);

  // Hỗ trợ tua
  const ar = r.headers.get('accept-ranges') || 'bytes';
  res.set('Accept-Ranges', ar);

  // Truyền status gốc (200 hoặc 206)
  res.status(r.status);

  // Stream body
  if (r.body) {
    // Node 20: r.body là web ReadableStream — convert sang Node stream
    Readable.fromWeb(r.body).pipe(res);
  } else {
    res.end();
  }
});

const PORT = process.env.PORT || 3004;
(async () => {
  try {
    await sequelize.sync();
    console.log('DB synced');
  } catch (e) {
    console.error('DB sync failed', e);
  }
})();
server.listen(PORT, () => console.log('Server listening on', PORT));
