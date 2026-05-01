class PcmDownsampleProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.frameSamples = 320;
    this._pending = [];
    this._pendingLength = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const mono = input[0];
    const ratio = sampleRate / this.targetSampleRate;
    const outputLength = Math.floor(mono.length / ratio);
    const downsampled = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i += 1) {
      const sourceIndex = Math.floor(i * ratio);
      downsampled[i] = mono[sourceIndex];
    }

    this._pending.push(downsampled);
    this._pendingLength += downsampled.length;

    while (this._pendingLength >= this.frameSamples) {
      const frame = new Int16Array(this.frameSamples);
      let written = 0;

      while (written < this.frameSamples) {
        const chunk = this._pending[0];
        const needed = this.frameSamples - written;
        const take = Math.min(needed, chunk.length);

        for (let i = 0; i < take; i += 1) {
          const sample = Math.max(-1, Math.min(1, chunk[i]));
          frame[written + i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        }

        written += take;

        if (take === chunk.length) {
          this._pending.shift();
        } else {
          this._pending[0] = chunk.slice(take);
        }
      }

      this._pendingLength -= this.frameSamples;
      this.port.postMessage(frame.buffer, [frame.buffer]);
    }

    return true;
  }
}

registerProcessor("pcm-downsample-processor", PcmDownsampleProcessor);

