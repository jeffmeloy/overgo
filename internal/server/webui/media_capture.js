// One capture lifetime per composer. Captured files enter its existing intake;
// this surface does not persist bytes or make inference requests.
(() => {
  'use strict';
  const el = overgo.el;
  // clipSliceMS: the recorder hands over a chunk this often, so the byte bound is checked while recording.
  const clipSliceMS = 1000;
  overgo.mediaCapture = function (options) {
    let dialog, session, file, objectURL, kind, state, previousFocus;
    let heading, status, preview, choices, actions, device, record, stop, photo, retake, attach;
    let serial = 0;
    const accepted = () => options.accept() || [];
    const button = (text, action) => el('button', { type: 'button', class: 'btn alt', text, onclick: action });
    const current = s => dialog && session === s && s.serial === serial;
    function release(s = session) {
      if (!s) return;
      if (session === s) session = null;
      if (s.node) { s.node.onprocessorerror = null; s.node.port.onmessage = null; s.node.port.close(); s.node.disconnect(); }
      if (s.source) s.source.disconnect();
      if (s.context) {
        s.context.onstatechange = null;
        s.context.close().catch(error => {
          // Late permission completion can release an already-closed context.
          // Other cleanup failures reach the shell's error collector.
          if (error.name !== 'InvalidStateError') throw error;
        });
      }
      if (s.recorder) { s.recorder.ondataavailable = s.recorder.onstop = s.recorder.onerror = null; if (s.recorder.state !== 'inactive') s.recorder.stop(); s.recorder = null; }
      if (s.stream) for (const track of s.stream.getTracks()) { track.onended = null; track.stop(); }
      if (s.video) { s.video.srcObject = null; s.video.removeAttribute('src'); }
    }
    function clearFile() {
      file = null;
      if (objectURL) URL.revokeObjectURL(objectURL);
      objectURL = null;
    }
    function close() {
      serial++;
      release(); clearFile();
      if (dialog) { const old = dialog; dialog = null; old.close(); old.remove(); }
      if (previousFocus && previousFocus.isConnected) previousFocus.focus();
      previousFocus = null;
    }
    function render(message = '') {
      if (!dialog) return;
      dialog.dataset.state = state;
      dialog.dataset.kind = kind;
      status.textContent = message;
      status.setAttribute('role', state === 'error' ? 'alert' : 'status');
      status.hidden = !message;
      choices.hidden = !!kind;
      actions.hidden = !kind;
      device.parentElement.hidden = !kind || state === 'ready';
      device.disabled = ['requesting', 'recording', 'finishing'].includes(state);
      // The record control serves audio (from idle) and clips (from the camera preview); a photo has its own control.
      const recordStates = kind === 'video' ? ['preview', 'error'] : ['idle', 'error'];
      record.hidden = (kind !== 'audio' && kind !== 'video') || !recordStates.includes(state);
      record.disabled = !recordStates.includes(state);
      record.textContent = recordLabel[kind === 'video' && state === 'error' ? 'retry' : kind];
      stop.hidden = (kind !== 'audio' && kind !== 'video') || state !== 'recording';
      photo.hidden = kind !== 'image' || state === 'ready';
      photo.disabled = state !== 'preview';
      retake.hidden = attach.hidden = state !== 'ready';
      attach.textContent = attachLabel[kind] || 'Attach';
    }
    const recordLabel = { audio: 'Record', video: 'Record clip', retry: 'Retry camera' };
    const attachLabel = { image: 'Attach photo', video: 'Attach clip', audio: 'Attach recording' };
    // ---- clips: the browser's recorder produces its own containers; a clip records only in one the
    // served input accepts, and a container is never renamed to another. ----
    const recorderTypes = () => ['video/mp4', 'video/webm'].filter(type => window.MediaRecorder && MediaRecorder.isTypeSupported(type));
    const clipType = () => accepted().find(type => type.startsWith('video/') && recorderTypes().includes(type)) || '';
    async function startClip() {
      const s = begin();
      try {
        s.stream = await navigator.mediaDevices.getUserMedia({ video: constraints(), audio: true });
        if (!current(s)) { release(s); return; }
        watchTracks(s); listDevices(s);
        s.video = el('video', { playsinline: '', autoplay: '', 'aria-label': 'Camera preview' });
        s.video.muted = true; s.video.srcObject = s.stream;
        preview.replaceChildren(s.video);
        const shown = () => { if (current(s) && state !== 'recording') { state = 'preview'; render('Camera preview. Record a clip when ready.'); } };
        s.video.addEventListener('loadeddata', shown);
        await s.video.play();
        if (current(s) && s.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) shown();
      } catch (error) { failure(error, s); }
    }
    function recordClip() {
      if (state === 'error') { startClip(); return; }
      const s = session, type = clipType(), limit = options.media.max_media_bytes;
      if (!s || state !== 'preview' || !type) return;
      if (!(limit > 0)) { failure(new Error('No usable clip byte limit is declared by the server.'), s); return; }
      try {
        s.recorder = new MediaRecorder(s.stream, { mimeType: type });
      } catch (error) { failure(error, s); return; }
      s.bytes = 0; s.chunks = []; s.started = Date.now();
      s.recorder.ondataavailable = event => {
        if (!current(s) || !event.data.size) return;
        s.bytes += event.data.size; s.chunks.push(event.data);
        if (s.bytes > limit) { s.recorder.onstop = null; s.recorder.stop(); failure(new Error('Clip exceeded the recording limit. Retake a shorter clip.'), s); return; }
        render('Recording · ' + Math.floor((Date.now() - s.started) / 1000) + 's · ' + overgo.fmt.bytes(s.bytes));
      };
      s.recorder.onerror = () => failure(new Error('Clip recording failed. Retry or attach a video file.'), s);
      s.recorder.onstop = () => {
        if (!current(s)) return;
        if (!s.bytes) { failure(new Error('No video was recorded. Try again.'), s); return; }
        ready(new Blob(s.chunks, { type }), 'clip.' + (type === 'video/mp4' ? 'mp4' : 'webm'), s);
      };
      s.recorder.start(clipSliceMS);
      state = 'recording'; render('Recording…'); stop.focus();
    }
    function stopClip() {
      if (!session || !session.recorder || state !== 'recording') return;
      state = 'finishing'; render('Finishing clip…');
      session.recorder.stop();
    }
    function failure(error, s) {
      if (s && !current(s)) { release(s); return; }
      if (!dialog) return;
      serial++; release(); clearFile(); preview.replaceChildren();
      const name = error && error.name;
      const message = name === 'NotAllowedError' ? 'Access was denied. Allow microphone or camera access in your browser, then retry.' :
        name === 'NotFoundError' ? 'No matching microphone or camera is available. Connect one, then retry.' :
        name === 'NotReadableError' ? 'The microphone or camera could not be opened. Check the device and retry.' :
        name === 'OverconstrainedError' ? 'That device is unavailable. Choose another device and retry.' :
        error.message || String(error);
      state = 'error'; render(message);
      if (kind === 'image') { photo.disabled = false; photo.textContent = 'Retry camera'; }
    }
    function begin() {
      serial++; release(); clearFile(); preview.replaceChildren();
      const s = { serial, chunks: [], samples: 0 };
      session = s; state = 'requesting'; render('Waiting for device permission…');
      return s;
    }
    function watchTracks(s) {
      for (const track of s.stream.getTracks()) track.onended = () => failure(new Error('The device disconnected. Reconnect it and retry.'), s);
    }
    async function listDevices(s) {
      if (!navigator.mediaDevices.enumerateDevices) return;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (!current(s)) return;
        const type = kind === 'audio' ? 'audioinput' : 'videoinput';
        const selected = device.value;
        const defaults = kind === 'audio' ? [el('option', { value: '', text: 'Default microphone' })] :
          [el('option', { value: 'user', text: 'Front camera' }), el('option', { value: 'environment', text: 'Rear camera' })];
        device.replaceChildren(...defaults, ...devices.filter(entry => entry.kind === type && entry.deviceId).map((entry, index) =>
          el('option', { value: 'device:' + entry.deviceId, text: entry.label || (kind === 'audio' ? 'Microphone ' : 'Camera ') + (index + 1) })));
        device.value = selected;
      } catch (error) {
        if (current(s)) render('Could not list other devices. You can still use the current device. ' + overgo.friendlyError(error));
      }
    }
    function constraints() {
      if (device.value.startsWith('device:')) return { deviceId: { exact: device.value.slice('device:'.length) } };
      return kind !== 'audio' ? { facingMode: { ideal: device.value || 'user' } } : true;
    }
    function ready(blob, name, s, message = '') {
      if (!current(s)) return;
      const limit = kind === 'image' ? options.media.max_image_bytes : options.media.max_media_bytes;
      if (!blob || !blob.size || !limit || blob.size > limit || !accepted().includes(blob.type)) {
        failure(new Error('The capture is empty, exceeds the file limit, or is no longer accepted here. Retry with a smaller capture.'), s); return;
      }
      release(s); clearFile();
      file = new File([blob], name, { type: blob.type });
      objectURL = URL.createObjectURL(file);
      preview.replaceChildren(kind === 'image' ? el('img', { src: objectURL, alt: 'Captured photo' }) : kind === 'video' ? el('video', { src: objectURL, controls: '', playsinline: '' }) : el('audio', { src: objectURL, controls: '' }));
      state = 'ready'; render(message || overgo.fmt.bytes(file.size) + ' · Preview before attaching.');
      attach.focus();
    }
    function wav(s) {
      // RIFF/WAVE PCM: mono, signed 16-bit little-endian samples.
      const sampleBytes = Int16Array.BYTES_PER_ELEMENT;
      const headerBytes = 44;
      const output = new ArrayBuffer(headerBytes + s.samples * sampleBytes);
      const view = new DataView(output);
      const tag = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
      tag(0, 'RIFF'); view.setUint32(4, output.byteLength - 8, true); tag(8, 'WAVE'); tag(12, 'fmt ');
      view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
      view.setUint32(24, s.context.sampleRate, true); view.setUint32(28, s.context.sampleRate * sampleBytes, true);
      view.setUint16(32, sampleBytes, true); view.setUint16(34, sampleBytes * 8, true);
      tag(36, 'data'); view.setUint32(40, s.samples * sampleBytes, true);
      let offset = headerBytes;
      for (const chunk of s.chunks) for (const value of chunk) { view.setInt16(offset, value, true); offset += sampleBytes; }
      return new Blob([output], { type: 'audio/wav' });
    }
    async function startAudio() {
      const s = begin();
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext || !window.AudioWorkletNode) throw new Error('This browser cannot record WAV audio. Attach an audio file instead.');
        const declaration = options.audio?.();
        const format = declaration?.format;
        if (declaration && (!format || !Number.isSafeInteger(format.sample_rate) || format.sample_rate <= 0 || format.channels !== 1 || format.encoding !== 'pcm-f32le' || !Number.isSafeInteger(declaration.maximum_encoded_bytes) || declaration.maximum_encoded_bytes <= 0 || !Number.isSafeInteger(declaration.maximum_samples) || declaration.maximum_samples <= 0)) {
          throw new Error('This model requires an audio format this recorder cannot produce. Attach a compatible audio file.');
        }
        const byteLimit = Math.min(options.media.max_media_bytes, declaration?.maximum_encoded_bytes ?? Infinity);
        const maxSamples = Math.min(Math.floor((byteLimit - 44) / Int16Array.BYTES_PER_ELEMENT), declaration?.maximum_samples ?? Infinity);
        if (!(maxSamples > 0)) throw new Error('No usable audio byte limit is declared by the server.');
        try { s.context = new AudioContext(format ? { sampleRate: format.sample_rate } : undefined); }
        catch (error) { if (format) throw new Error('This browser cannot record at ' + format.sample_rate + ' Hz. Attach a compatible audio file.', { cause: error }); throw error; }
        if (format && s.context.sampleRate !== format.sample_rate) throw new Error('This browser cannot record at ' + format.sample_rate + ' Hz. Attach a compatible audio file.');
        await s.context.resume();
        if (!current(s)) return;
        s.stream = await navigator.mediaDevices.getUserMedia({ audio: constraints(), video: false });
        if (!current(s)) { release(s); return; }
        watchTracks(s); listDevices(s);
        await s.context.audioWorklet.addModule('/pcm_capture.js');
        if (!current(s)) return;
        s.node = new AudioWorkletNode(s.context, 'pcm-capture', { processorOptions: { maxSamples, chunkSamples: Math.min(maxSamples, s.context.sampleRate) } });
        s.node.port.onmessage = event => {
          if (!current(s)) return;
          if (event.data.samples) {
            const samples = event.data.samples;
            if (s.samples + samples.length > maxSamples) { failure(new Error('Audio exceeded the recording limit.'), s); return; }
            s.chunks.push(samples); s.samples += samples.length;
            render('Recording · ' + Math.floor(s.samples / s.context.sampleRate) + 's · ' + overgo.fmt.bytes(s.samples * Int16Array.BYTES_PER_ELEMENT));
          }
          if (event.data.done) {
            if (!s.samples) { failure(new Error('No audio was recorded. Try again.'), s); return; }
            ready(wav(s), 'recording.wav', s, s.samples === maxSamples ? 'Recording limit reached. Preview before attaching.' : '');
          }
        };
        s.node.onprocessorerror = () => failure(new Error('Audio recording failed. Retry or attach an audio file.'), s);
        s.context.onstatechange = () => { if (current(s) && s.context.state !== 'running') failure(new Error('Audio recording was interrupted. Retry the recording.'), s); };
        s.source = s.context.createMediaStreamSource(s.stream);
        s.source.connect(s.node); s.node.connect(s.context.destination);
        state = 'recording'; render('Recording…'); stop.focus();
      } catch (error) { failure(error, s); }
    }
    function stopAudio() {
      if (!session || !session.node || state !== 'recording') return;
      state = 'finishing'; render('Finishing recording…');
      session.node.port.postMessage('stop');
    }
    async function startCamera() {
      const s = begin();
      photo.textContent = 'Take photo';
      try {
        s.stream = await navigator.mediaDevices.getUserMedia({ video: constraints(), audio: false });
        if (!current(s)) { release(s); return; }
        watchTracks(s); listDevices(s);
        s.video = el('video', { playsinline: '', autoplay: '', 'aria-label': 'Camera preview' });
        s.video.muted = true; s.video.srcObject = s.stream;
        preview.replaceChildren(s.video);
        s.video.addEventListener('loadeddata', () => { if (current(s)) { state = 'preview'; render('Camera preview. Take a photo when ready.'); } });
        await s.video.play();
        if (current(s) && s.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) { state = 'preview'; render('Camera preview. Take a photo when ready.'); }
      } catch (error) { failure(error, s); }
    }
    async function takePhoto() {
      if (state === 'error') { startCamera(); return; }
      const s = session;
      if (!s || state !== 'preview' || !s.video.videoWidth || !s.video.videoHeight) return;
      const bounds = options.media;
      const width = s.video.videoWidth, height = s.video.videoHeight;
      const scale = Math.min(1, bounds.max_image_dimension / width, bounds.max_image_dimension / height, Math.sqrt(bounds.max_image_pixels / (width * height)));
      if (!(scale > 0)) { failure(new Error('No usable image bounds are declared by the server.'), s); return; }
      let canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.floor(width * scale)); canvas.height = Math.max(1, Math.floor(height * scale));
      state = 'finishing'; render('Preparing photo…');
      try {
        canvas.getContext('2d').drawImage(s.video, 0, 0, canvas.width, canvas.height);
        // A valid decoded image can still exceed the encoded byte limit.
        // Resize the frozen frame, never a later camera frame. Each retry
        // strictly reduces its dimensions and stops at one pixel.
        while (current(s)) {
          const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
          if (!current(s)) return;
          if (!blob || blob.size <= bounds.max_image_bytes || (canvas.width === 1 && canvas.height === 1)) {
            ready(blob, 'photo.png', s); return;
          }
          const ratio = Math.sqrt(bounds.max_image_bytes / blob.size);
          if (!(ratio > 0)) throw new Error('No usable image byte limit is declared by the server.');
          const smaller = document.createElement('canvas');
          smaller.width = Math.max(1, Math.floor(canvas.width * ratio));
          smaller.height = Math.max(1, Math.floor(canvas.height * ratio));
          smaller.getContext('2d').drawImage(canvas, 0, 0, smaller.width, smaller.height);
          canvas = smaller;
        }
      } catch (error) { failure(error, s); }
    }
    function choose(next) {
      serial++; release(); clearFile(); preview.replaceChildren();
      kind = next; state = 'idle';
      heading.textContent = kind === 'audio' ? 'Record audio' : kind === 'image' ? 'Take a photo' : kind === 'video' ? 'Record a clip' : 'Attach';
      device.replaceChildren(...(kind === 'image' || kind === 'video' ? [el('option', { value: 'user', text: 'Front camera' }), el('option', { value: 'environment', text: 'Rear camera' })] : [el('option', { value: '', text: 'Default microphone' })]));
      render(kind === 'audio' ? 'Record, preview, then attach. Recording stays on this device until you attach it.' : '');
      if (kind === 'image') startCamera();
      else if (kind === 'video') startClip();
      else if (kind === 'audio') record.focus();
    }
    function open() {
      close(); previousFocus = document.activeElement;
      kind = ''; state = 'idle';
      heading = el('h2', { text: 'Attach' });
      status = el('p', { class: 'note', role: 'status', hidden: true });
      preview = el('div', { class: 'capture-preview' });
      const files = button('Choose files', () => { close(); options.files(); });
      const audio = button('Record audio', () => choose('audio'));
      const camera = button('Take a photo', () => choose('image'));
      const clip = button('Record a clip', () => choose('video'));
      const secure = window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;
      const unavailable = !window.isSecureContext ? 'Microphone and camera access require HTTPS or localhost. File attachments are still available.' :
        'This browser does not provide microphone or camera access. You can still choose files.';
      audio.hidden = !accepted().some(type => type.startsWith('audio/'));
      camera.hidden = !accepted().some(type => type.startsWith('image/'));
      clip.hidden = !accepted().some(type => type.startsWith('video/'));
      audio.disabled = !secure || !accepted().includes('audio/wav');
      camera.disabled = !secure || !accepted().includes('image/png');
      clip.disabled = !secure || !clipType();
      const audioRefusal = secure ? 'This input does not accept WAV audio.' : unavailable;
      const cameraRefusal = secure ? 'This input does not accept photos.' : unavailable;
      const acceptedVideo = accepted().filter(type => type.startsWith('video/'));
      const clipRefusal = secure ? 'This browser records ' + (recorderTypes().join(', ') || 'no clip format') + '; this input accepts ' + (acceptedVideo.join(', ') || 'no video') + '. Choose a video file instead.' : unavailable;
      audio.title = audio.disabled ? audioRefusal : '';
      camera.title = camera.disabled ? cameraRefusal : '';
      clip.title = clip.disabled ? clipRefusal : '';
      choices = el('div', { class: 'capture-choices' }, files, audio, camera, clip);
      device = el('select', { class: 'text', 'aria-label': 'Capture device', onchange: () => { if (kind === 'image') startCamera(); else if (kind === 'video') startClip(); } });
      record = button('Record', () => { if (kind === 'video') recordClip(); else startAudio(); }); stop = button('Stop recording', () => { if (kind === 'video') stopClip(); else stopAudio(); }); photo = button('Take photo', takePhoto);
      retake = button('Retake', () => { if (kind === 'image') startCamera(); else if (kind === 'video') startClip(); else { clearFile(); preview.replaceChildren(); state = 'idle'; render('Ready to record again.'); record.focus(); } });
      attach = button('Attach', () => { if (file && accepted().includes(file.type)) { const result = file; close(); options.addFile(result); } });
      for (const primary of [record, stop, photo, attach]) primary.classList.remove('alt');
      actions = el('div', { class: 'row capture-actions' }, button('Back', () => choose('')), record, stop, photo, retake, attach);
      dialog = el('dialog', { class: 'surface-dialog capture-dialog', 'aria-label': 'Attach files or capture media', oncancel: event => { event.preventDefault(); close(); } },
        el('div', { class: 'dialog-heading' }, heading, button('Close', close)), choices,
        el('label', { class: 'setting-field' }, 'Device', device), preview, status, actions);
      dialog.addEventListener('close', () => { if (dialog && !dialog.open) close(); });
      document.body.appendChild(dialog); render(); dialog.showModal();
      const declined = [audio, camera, clip].filter(button => !button.hidden && button.disabled);
      if (declined.length) { status.hidden = false; status.textContent = [...new Set(declined.map(button => button.title))].join(' '); }
    }
    const hidden = () => { if (document.hidden) close(); };
    window.addEventListener('pagehide', close);
    window.addEventListener('overgo-panel-change', close);
    document.addEventListener('visibilitychange', hidden);
    return { open, close, dispose() {
      close(); window.removeEventListener('pagehide', close);
      window.removeEventListener('overgo-panel-change', close);
      document.removeEventListener('visibilitychange', hidden);
    } };
  };
})();
