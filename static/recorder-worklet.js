class RecorderProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const targetRate = (options.processorOptions && options.processorOptions.targetSampleRate) || 16000;
    this.ratio = sampleRate / targetRate; // `sampleRate` is a global in AudioWorkletGlobalScope
    this.buffer = [];
    this.chunkSize = 1600; // ~100ms at 16kHz, matches the server's expected chunk size
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const channel = input[0];
      for (let i = 0; i < channel.length; i += this.ratio) {
        this.buffer.push(channel[Math.floor(i)] || 0);
      }
      while (this.buffer.length >= this.chunkSize) {
        const slice = this.buffer.splice(0, this.chunkSize);
        const pcm16 = new Int16Array(slice.length);
        for (let j = 0; j < slice.length; j++) {
          const s = Math.max(-1, Math.min(1, slice[j]));
          pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
      }
    }
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
