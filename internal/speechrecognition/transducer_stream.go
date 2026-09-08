package speechrecognition

import (
	"context"
	"errors"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
)

type transducerStream struct {
	encoder       *encoderStream
	subsampling   subsamplingState
	bytes         uint64
	failed, final bool
}

func (t *Transducer) streamGeometry(visit func(int)) (first, following int, bytes uint64, err error) {
	factor, bands, channels := 1, t.binding.Bands, 1
	var elements uint64
	for _, layer := range t.subsampling {
		b := layer.binding
		k := int(layer.weight.Shape.Dims[1])
		stride := int(b.Stride[1])
		if k < stride || int(b.Padding[2]) != k-1 || int(b.Padding[3]) != stride-1 {
			return 0, 0, 0, errors.New("transducer stream: subsampling is not causal")
		}
		plane, ok := checked.MulInt(channels, bands)
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: cache plane overflows")
		}
		count, ok := checked.MulInt(k-stride, plane)
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: cache extent overflows")
		}
		if visit != nil {
			visit(count)
		}
		elements, ok = checked.Add64(elements, uint64(count))
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: cache sum overflows")
		}
		factor, ok = checked.MulInt(factor, stride)
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: stride extent overflows")
		}
		frequency, ok := checked.AddInt(bands, int(b.Padding[0]), int(b.Padding[1]))
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: frequency extent overflows")
		}
		bands = (frequency-int(layer.weight.Shape.Dims[0]))/int(b.Stride[0]) + 1
		channels = int(layer.weight.Shape.Dims[3])
	}
	chunk := t.encoder.blocks[0].attention.block
	for _, block := range t.encoder.blocks {
		a, c := block.attention, block.convolution
		if a.position.in == 0 || a.block != chunk || a.leftContext%chunk != 0 || !c.causal || c.stride != 1 {
			return 0, 0, 0, errors.New("transducer stream: incompatible attention partition or convolution")
		}
		width, ok := checked.MulInt(2, a.query.out)
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: key/value width overflows")
		}
		attention, ok := checked.MulInt(a.leftContext, width)
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: attention cache overflows")
		}
		convolution, ok := checked.MulInt(c.kernelSize-1, c.channels)
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: convolution cache overflows")
		}
		if visit != nil {
			visit(attention / 2) // Key and value have the same declared width.
			visit(attention / 2)
			visit(convolution)
		}
		elements, ok = checked.Add64(elements, uint64(attention), uint64(convolution))
		if !ok {
			return 0, 0, 0, errors.New("transducer stream: cache sum overflows")
		}
	}
	if elements > t.memoryBytes/binaryschema.Uint32Bytes {
		return 0, 0, 0, errors.New("transducer stream: caches exceed byte budget")
	}
	first, ok := checked.MulInt(factor, chunk-1)
	if !ok {
		return 0, 0, 0, errors.New("transducer stream: first chunk overflows")
	}
	first, ok = checked.AddInt(first, 1)
	if !ok {
		return 0, 0, 0, errors.New("transducer stream: first chunk overflows")
	}
	following, ok = checked.MulInt(factor, chunk)
	if !ok {
		return 0, 0, 0, errors.New("transducer stream: chunk overflows")
	}
	return first, following, elements * binaryschema.Uint32Bytes, nil
}

// RecognizeChunk consumes one exact declared mel chunk, preserving bounded
// convolution, attention and recurrent state. A final chunk must already be
// explicitly padded to the declared size. Results contain only new emissions.
// A failed or finalized stream cannot resume; use a new workspace to restart.
func (t *Transducer) RecognizeChunk(ctx context.Context, features []float32, frames int, final bool, w *TransducerWorkspace, observe *TransducerObserver) (TransducerResult, error) {
	if t == nil || t.encoder == nil || len(t.encoder.blocks) == 0 || len(t.recurrent) == 0 || ctx == nil || w == nil || w.owner != nil && w.owner != t {
		return TransducerResult{}, errors.New("transducer stream: invalid invocation")
	}
	if err := ctx.Err(); err != nil {
		return TransducerResult{}, err
	}
	first, following, cacheBytes, err := t.streamGeometry(nil)
	if err != nil {
		return TransducerResult{}, err
	}
	expected := first
	if w.stream != nil {
		if w.stream.failed || w.stream.final {
			return TransducerResult{}, errors.New("transducer stream: failed or finalized")
		}
		expected = following
		if _, ok := checked.AddInt(w.stream.encoder.frames, t.encoder.blocks[0].attention.block); !ok {
			return TransducerResult{}, errors.New("transducer stream: frame position overflows")
		}
		emissions, ok := checked.MulInt(t.encoder.blocks[0].attention.block, t.binding.MaxSymbolsPerFrame)
		if !ok {
			return TransducerResult{}, errors.New("transducer stream: emission extent overflows")
		}
		if _, ok := checked.AddInt(w.steps, emissions); !ok {
			return TransducerResult{}, errors.New("transducer stream: emission position overflows")
		}
	} else if w.owner != nil {
		return TransducerResult{}, errors.New("transducer stream: offline workspace cannot start streaming")
	}
	count, ok := checked.MulInt(frames, t.binding.Bands)
	if !ok || frames != expected || len(features) != count {
		return TransducerResult{}, errors.New("transducer stream: wrong mel chunk geometry")
	}
	if w.stream == nil {
		if cacheBytes > t.memoryBytes-t.weightBytes {
			return TransducerResult{}, errors.New("transducer stream: cache exceeds remaining budget")
		}
		w.stream = &transducerStream{encoder: t.encoder.newStream(), subsampling: subsamplingState{tails: make([][]float32, len(t.subsampling))}, bytes: cacheBytes}
	}
	w.stream.failed = true
	result, err := t.recognize(ctx, features, frames, w, observe)
	if err != nil {
		return TransducerResult{}, err
	}
	w.stream.failed = false
	w.stream.final = final
	return result, nil
}
