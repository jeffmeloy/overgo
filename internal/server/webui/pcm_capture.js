// Capture mono PCM on the audio rendering thread. The caller supplies the
// sample bound derived from the server's WAV byte limit.
class PCMCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.remaining = options.processorOptions.maxSamples;
    this.chunkSamples = options.processorOptions.chunkSamples;
    this.buffer = new Int16Array(Math.min(this.chunkSamples, this.remaining));
    this.used = 0;
    this.stopped = false;
    this.port.onmessage = event => {
      if (event.data === 'stop' && !this.stopped) this.finish();
    };
  }
  finish() {
    this.flush();
    this.stopped = true;
    this.port.postMessage({ done: true });
  }
  flush() {
    if (!this.used) return;
    const samples = this.used === this.buffer.length ? this.buffer : this.buffer.slice(0, this.used);
    this.port.postMessage({ samples }, [samples.buffer]);
    this.buffer = new Int16Array(Math.min(this.chunkSamples, this.remaining));
    this.used = 0;
  }
  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    const count = Math.min(channels[0].length, this.remaining);
    for (let i = 0; i < count; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i];
      value = Math.max(-1, Math.min(1, value / channels.length));
      this.buffer[this.used++] = Math.round(value * (value < 0 ? 0x8000 : 0x7fff));
      this.remaining--;
      if (this.used === this.buffer.length) this.flush();
    }
    if (!this.remaining) this.finish();
    // Outputs stay silent; microphone audio is never routed to the speakers.
    return !this.stopped;
  }
}
registerProcessor('pcm-capture', PCMCapture);
