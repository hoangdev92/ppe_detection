<template>
  <div style="padding:12px; max-width:820px; margin:auto;">
    <h3>YOLOv11 Realtime Demo</h3>

    <div style="display:flex; gap:10px; align-items:center;">
      <button @click="useWebcam" :disabled="mode==='webcam'">Use Webcam</button>
      <input type="file" accept="video/*" @change="onFileSelected"/>
      <input v-model="videoUrl" placeholder="Video URL (mp4, http)" style="width:320px"/>
      <button @click="useUrl">Use URL</button>
      <label style="margin-left:8px">
        FPS:
        <input type="number" v-model.number="fps" min="1" max="15" style="width:64px"/>
      </label>
      <label>
        SendSize:
        <select v-model.number="sendWidth">
          <option :value="320">320</option>
          <option :value="480">480</option>
          <option :value="640">640</option>
        </select>
      </label>
      <span style="margin-left:auto">client: {{clientId || '—'}}</span>
    </div>

    <div style="position:relative; margin-top:10px;">
      <video ref="video" autoplay playsinline controls width="640" height="480" style="background:#000;"></video>
      <canvas ref="overlay" width="640" height="480" style="position:absolute; left:0; top:0; width:640px; height:480px; pointer-events:none;"></canvas>
    </div>

    <div style="margin-top:8px;">
      <button @click="start" :disabled="running">Start</button>
      <button @click="stop" :disabled="!running">Stop</button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      ws: null,
      clientId: null,
      running: false,
      fps: 5,
      sendWidth: 320,
      sendHeight: 0, // calculated from video aspect
      timer: null,
      mode: null, // 'webcam' | 'file' | 'url'
      fileUrl: null,
      videoUrl: '',
      pendingMeta: false,
    };
  },
  unmounted() {
    window.removeEventListener('resize', this.prepareSendSize())
  },
  mounted() {
    this.setupWs();
    window.addEventListener('resize', this.prepareSendSize())
  },
  methods: {
    setupWs() {
      console.log("✅ Setting up WebSocket");
      const loc = window.location;
      const wsProto = loc.protocol === 'https:' ? 'wss' : 'ws';
      // assume same host/port as server; adjust if server on other port
      const url = `${wsProto}://${loc.hostname}:3004`;
      this.ws = new WebSocket(url);
      this.ws.binaryType = 'arraybuffer';
      this.ws.onopen = () => {
        console.log("✅ WebSocket opened");
      };
      this.ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'hello' && msg.clientId) {
            this.clientId = msg.clientId;
          } else if (msg.type === 'inference' && msg.boxes) {
            if (!this.running) return; // ignore late frames after stop
            this.drawBoxes(msg.boxes);
          } else if (msg.type === 'violation') {
            const txt = `Thiếu PPE: ${msg.missingItems.join(', ')} (tracks: ${Array.isArray(msg.tracks) ? msg.tracks.join(',') : '-'})`;
            console.warn(txt);
            // alert(txt); // hoặc thay bằng toast UI
          }
        } catch(e) {
          console.warn('non-json ws msg', e);
        }
      };
      this.ws.onclose = () => console.log('ws closed');
    },

    useWebcam() {
      this.mode = 'webcam';
    },
    async useUrl() {
      if (!this.videoUrl) return alert('Nhập URL video trước');
      this.mode = 'url';
      const v = this.$refs.video;
      v.srcObject = null;

      // Quan trọng: đặt trước khi gán src để tránh tainted canvas
      v.crossOrigin = 'anonymous';

      // Dùng proxy cùng-origin (server Node đang chạy cổng 3004)
      const backend = `${location.protocol}//${location.hostname}:3004`;
      const proxied = `${backend}/proxy?url=` + encodeURIComponent(this.videoUrl);
      console.log('[proxy]', proxied);
      const isHls = /\.m3u8($|\?)/i.test(this.videoUrl);
      if (isHls) {
        // Chrome/Firefox cần hls.js
        if (window.Hls) {
          if (this._hls) { this._hls.destroy(); this._hls = null; }
          this._hls = new window.Hls({ lowLatencyMode: true });
          this._hls.loadSource(proxied);
          this._hls.attachMedia(v);
          await new Promise((resolve) => {
            this._hls.on(window.Hls.Events.MANIFEST_PARSED, resolve);
          });
          v.play().catch(e => console.warn(e));
        } else if (v.canPlayType('application/vnd.apple.mpegurl')) {
          // Safari phát được HLS native
          v.src = proxied;
          await v.play().catch(e => console.warn(e));
        } else {
          alert('Cần hls.js để phát HLS trên trình duyệt này');
          return;
        }
      } else {
        // MP4/WebM thông thường
        v.src = proxied;
        await v.play().catch(e => console.warn(e));
      }
      // v.src = proxied;

      // await v.play().catch(e => console.warn(e));
      this.prepareSendSize();
    },
    onFileSelected(e) {
      const f = e.target.files[0];
      if (!f) return;
      this.mode = 'file';
      if (this.fileUrl) URL.revokeObjectURL(this.fileUrl);
      this.fileUrl = URL.createObjectURL(f);
      const v = this.$refs.video;
      v.srcObject = null;
      v.src = this.fileUrl;
      v.loop = true;
      v.play();
      this.prepareSendSize();
    },

    async start() {
      const v = this.$refs.video;
      if (this.mode === 'webcam') {
        try {
          const constraints = { video: { width: 640, height: 480 } };
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          v.srcObject = stream;
          await v.play();
        } catch (err) {
          alert('Không thể mở webcam: ' + err.message);
          return;
        }
      } else if (this.mode === null) {
        alert('Chọn webcam, upload video, hoặc nhập URL trước');
        return;
      }
      this.prepareSendSize();
      this.running = true;
      const interval = Math.round(1000 / Math.max(1, this.fps));
      const offscreen = document.createElement('canvas');
      offscreen.width = this.sendWidth;
      offscreen.height = this.sendHeight;
      const offctx = offscreen.getContext('2d');

      this.timer = setInterval(async () => {
        if (!this.running || this.ws.readyState !== WebSocket.OPEN) return;
        // draw resized frame
        offctx.drawImage(v, 0, 0, offscreen.width, offscreen.height);
        // convert to blob (jpeg) and send binary with meta
        offscreen.toBlob((blob) => {
          if (!blob) return;
          // send meta first (text)
          const meta = {
            type: 'frame_meta',
            clientId: this.clientId,
            width: offscreen.width,
            height: offscreen.height,
            size: blob.size
          };
          try {
            this.ws.send(JSON.stringify(meta));
            // then send binary blob
            this.ws.send(blob);
          } catch (e) {
            console.warn('ws send failed', e);
          }
        }, 'image/jpeg', 0.6);
      }, interval);
    },

    stop() {
      this.running = false;
      clearInterval(this.timer);
      const overlay = this.$refs.overlay;
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      setTimeout(() => {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }, 100);
    },

    prepareSendSize() {
      const v = this.$refs.video;
      const vw = v.videoWidth || 640;
      const vh = v.videoHeight || 480;
      const aspect = vw && vh ? vw / vh : (640/480);
      this.sendHeight = Math.round(this.sendWidth / aspect);

      // Canvas overlay phải khớp kích thước hiển thị của video
      const overlay = this.$refs.overlay;
      const dw = v.clientWidth || 640;
      const dh = v.clientHeight || 480;
      overlay.width = dw;
      overlay.height = dh;
    },

    drawBoxes(boxes) {
      if (!this.running) return; // prevent drawing when stopped
      const overlay = this.$refs.overlay;
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0,0,overlay.width, overlay.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'red';
      ctx.font = '16px Arial';
      ctx.fillStyle = 'red';

      // Tính content rect (khu vực video thật sự hiển thị bên trong khung overlay)
      const v = this.$refs.video;
      const vw = v.videoWidth || this.sendWidth;
      const vh = v.videoHeight || this.sendHeight;
      const dw = overlay.width;
      const dh = overlay.height;

      if (!vw || !vh || !dw || !dh) return;

      const scaleFit = Math.min(dw / vw, dh / vh);
      const contentW = vw * scaleFit;
      const contentH = vh * scaleFit;
      const offsetX = (dw - contentW) / 2;
      const offsetY = (dh - contentH) / 2;

      // Box đang ở hệ toạ độ sendWidth/sendHeight -> scale sang contentW/contentH rồi cộng offset
      const scaleX = contentW / this.sendWidth;
      const scaleY = contentH / this.sendHeight;

      boxes.forEach(b => {
        const x = offsetX + b.x1 * scaleX;
        const y = offsetY + b.y1 * scaleY;
        const w = (b.x2 - b.x1) * scaleX;
        const h = (b.y2 - b.y1) * scaleY;

        ctx.strokeRect(x, y, w, h);
        const label = `${b.class ?? 'obj'} ${(b.conf ?? 0).toFixed(2)}`;
        ctx.fillText(label, x + 2, y + 16);
      });
    }
  }
};
</script>
