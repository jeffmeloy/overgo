package speechrecognition

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/adaptertrain"
	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/scratch"
)

// Workspace owns reusable per-clip storage. Its zero value is usable. It must
// not be shared concurrently, between encoders, or reentered from an observer.
type Workspace struct {
	adapter                                                                            *adaptertrain.LinearCTC
	owner                                                                              *Encoder
	x, normalized, branch, expanded, query, key, value, attended, conv, logits, scores []float32
	relativeInput, relativeKey                                                         []float32
	reservedBytes                                                                      uint64 // Numeric state owned by a composing executor.
}

// Trace exposes borrowed row-major values at an encoder boundary. Block -1 is
// the input projection; other values identify zero-based block outputs before
// any posterior feedback. Values must not be retained without copying.
type Trace struct {
	Block, Frames, Width int
	Values               []float32
}

func (w *Workspace) buffers() []*[]float32 {
	return []*[]float32{&w.x, &w.normalized, &w.branch, &w.expanded, &w.query, &w.key, &w.value, &w.attended, &w.conv, &w.logits, &w.scores, &w.relativeInput, &w.relativeKey}
}

func (e *Encoder) workspace(w *Workspace, rows int) error {
	h := e.input.out
	ff, channels, query, attentionBlock := 0, 0, 0, 0
	projectedRows, logitsRows := rows, 0
	relativeInputWidth, relativeKeyWidth := 0, 0
	for i, b := range e.blocks {
		ff = max(ff, b.first.in.out, b.second.in.out, b.convolution.in.out)
		channels = max(channels, b.convolution.channels)
		query = max(query, b.attention.query.out)
		attentionBlock = max(attentionBlock, b.attention.block)
		if b.attention.position.in != 0 {
			relativeInputWidth = max(relativeInputWidth, b.attention.position.in)
			relativeKeyWidth = max(relativeKeyWidth, b.attention.position.out)
			attentionBlock = max(attentionBlock, min(rows, b.attention.leftContext)+min(rows, b.attention.block))
		}
		projectedRows /= b.convolution.stride
		if i+1 == e.feedbackAfter {
			logitsRows = projectedRows
		}
	}
	logitsRows = max(logitsRows, projectedRows)
	widths := [...]int{h, h, h, ff, query, query, query, query, channels, e.output.out, attentionBlock, relativeInputWidth, relativeKeyWidth}
	// Attention uses one reusable score row, not a quadratic retained trace.
	var counts [len(widths)]int
	bytes, valid := checked.Add64(e.weightBytes, w.reservedBytes)
	if !valid || bytes > e.memoryBytes {
		return errors.New("encoder composed state exceeds byte budget")
	}
	if w.adapter != nil {
		var ok bool
		bytes, ok = checked.Add64(bytes, w.adapter.StorageBytes())
		if !ok || bytes > e.memoryBytes {
			return errors.New("encoder adapter state exceeds byte budget")
		}
	}
	for i, buffer := range w.buffers() {
		n, ok := checked.MulInt(rows, widths[i])
		if buffer == &w.logits {
			n, ok = checked.MulInt(logitsRows, e.output.out)
		}
		if buffer == &w.scores {
			n, ok = widths[i], true
		}
		if buffer == &w.relativeInput || buffer == &w.relativeKey {
			positions, valid := checked.AddInt(rows, rows-1)
			if !valid {
				return errors.New("encoder relative position extent overflows")
			}
			n, ok = checked.MulInt(positions, widths[i])
		}
		if !ok {
			return errors.New("encoder workspace extent overflows")
		}
		counts[i] = n
		retained := max(cap(*buffer), n)
		if uint64(retained) > (e.memoryBytes-bytes)/binaryschema.Uint32Bytes {
			return errors.New("encoder workspace exceeds byte budget")
		}
		bytes += uint64(retained) * binaryschema.Uint32Bytes
	}
	for i, buffer := range w.buffers() {
		*buffer = scratch.Resize(*buffer, counts[i])
	}
	w.owner = e
	return nil
}

func (p linear) forward(dst, x []float32, rows int) {
	hostmath.LinearInputProjection(dst, nil, x, nil, p.weight, p.bias, rows, p.in, p.out)
}

func (e *Encoder) normalize(dst, x []float32, n norm, rows int) {
	hostmath.LayerNormInto(dst, x, n.weight, n.bias, rows, e.input.out, e.lnEpsilon)
}

func (e *Encoder) feedForward(ctx context.Context, x []float32, rows int, f feedForward, w *Workspace) error {
	e.normalize(w.normalized, x, f.norm, rows)
	f.in.forward(w.expanded, w.normalized, rows)
	if err := ctx.Err(); err != nil {
		return err
	}
	values := w.expanded[:rows*f.in.out]
	if e.activation == "silu" {
		hostmath.SiLUInPlace(values)
	} else {
		hostmath.GELUErfInPlace(values)
	}
	f.out.forward(w.branch, values, rows)
	for i := range x {
		x[i] += e.ffScale * w.branch[i]
	}
	return ctx.Err()
}

// Encode processes one complete, unpadded clip and returns borrowed final
// hidden features and output frame count. Project applies the frozen output.
// It does not accept padded batches or provide streaming semantics. Right padding inside attention blocks is
// excluded from the softmax, and convolution subsampling drops incomplete pools.
// Only requested boundary observations are retained by the caller.
func (e *Encoder) Encode(ctx context.Context, features []float32, frames int, w *Workspace, observe func(Trace) error) ([]float32, int, error) {
	return e.encode(ctx, features, frames, w, observe, nil)
}

func (e *Encoder) encode(ctx context.Context, features []float32, frames int, w *Workspace, observe func(Trace) error, stream *encoderStream) ([]float32, int, error) {
	if e == nil || len(e.blocks) == 0 || ctx == nil || w == nil || w.owner != nil && w.owner != e || frames <= 0 {
		return nil, 0, errors.New("encoder: invalid execution")
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	n, ok := checked.MulInt(frames, e.input.in)
	if !ok || len(features) != n {
		return nil, 0, errors.New("encoder: input shape differs")
	}
	outputFrames := frames
	for index, b := range e.blocks {
		positions := outputFrames
		if stream != nil {
			var valid bool
			positions, valid = checked.AddInt(positions, stream.attention[index].frames)
			if !valid {
				return nil, 0, errors.New("encoder: cached position extent overflows")
			}
		}
		if b.attention.position.in != 0 && positions > b.attention.maxPositions {
			return nil, 0, errors.New("encoder: declared position limit exceeded")
		}
		outputFrames /= b.convolution.stride
		if outputFrames == 0 {
			return nil, 0, errors.New("encoder: clip shorter than residual pooling extent")
		}
	}
	for _, buffer := range w.buffers() {
		if checked.SlicesOverlap(features, (*buffer)[:cap(*buffer)]) {
			return nil, 0, errors.New("encoder: input aliases workspace")
		}
	}
	for _, v := range features {
		if !checked.Finite32(v) {
			return nil, 0, errors.New("encoder: non-finite input")
		}
	}
	bufferRows := frames
	if stream != nil {
		for _, history := range stream.attention {
			bufferRows = max(bufferRows, frames+history.frames)
		}
		for _, b := range e.blocks {
			bufferRows = max(bufferRows, frames+b.convolution.kernelSize-1)
		}
	}
	if err := e.workspace(w, bufferRows); err != nil {
		return nil, 0, err
	}
	h := e.input.out
	x := w.x[:frames*h]
	e.input.forward(x, features, frames)
	if e.scaleInputByWidth {
		scale := float32(math.Sqrt(float64(h)))
		for i := range x {
			x[i] *= scale
		}
	}
	if err := emitTrace(ctx, observe, -1, frames, h, x); err != nil {
		return nil, 0, err
	}
	for i, b := range e.blocks {
		var history *attentionHistory
		var convHistory []float32
		if stream != nil {
			history, convHistory = &stream.attention[i], stream.convolution[i]
		}
		if err := e.feedForward(ctx, x, frames, b.first, w); err != nil {
			return nil, 0, err
		}
		e.normalize(w.normalized, x, b.attention.norm, frames)
		if err := b.attention.forward(ctx, w.normalized, frames, w, history); err != nil {
			return nil, 0, err
		}
		for j := range x {
			x[j] += w.branch[j]
		}
		e.normalize(w.normalized, x, b.convolution.norm, frames)
		if err := b.convolution.forward(ctx, w.normalized, frames, e.bnEpsilon, e.lnEpsilon, w, convHistory); err != nil {
			return nil, 0, err
		}
		stride := b.convolution.stride
		outFrames := frames / stride
		// Pool in place: every destination precedes its unread source group.
		for row := range outFrames {
			for col := range h {
				var sum float32
				for k := range stride {
					sum += x[(row*stride+k)*h+col]
				}
				x[row*h+col] = sum/float32(stride) + w.branch[row*h+col]
			}
		}
		frames = outFrames
		x = x[:frames*h]
		if err := e.feedForward(ctx, x, frames, b.second, w); err != nil {
			return nil, 0, err
		}
		e.normalize(x, x, b.out, frames)
		if err := emitTrace(ctx, observe, i, frames, h, x); err != nil {
			return nil, 0, err
		}
		if i+1 == e.feedbackAfter {
			e.output.forward(w.logits, x, frames)
			for row := range frames {
				hostmath.SoftmaxInPlace(w.logits[row*e.output.out : (row+1)*e.output.out])
			}
			e.feedback.forward(w.branch, w.logits, frames)
			for j := range x {
				x[j] += w.branch[j]
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	for _, v := range x {
		if !checked.Finite32(v) {
			return nil, 0, errors.New("encoder: non-finite output")
		}
	}
	if stream != nil {
		stream.frames += frames
	}
	return x, frames, nil
}

func emitTrace(ctx context.Context, observe func(Trace) error, block, frames, width int, values []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if observe != nil {
		if err := observe(Trace{block, frames, width, values}); err != nil {
			return fmt.Errorf("encoder observation: %w", err)
		}
	}
	return nil
}

func (a attention) forward(ctx context.Context, x []float32, rows int, w *Workspace, history *attentionHistory) error {
	a.query.forward(w.query, x, rows)
	if err := ctx.Err(); err != nil {
		return err
	}
	a.key.forward(w.key, x, rows)
	a.value.forward(w.value, x, rows)
	if a.position.in != 0 {
		return a.projectedRelative(ctx, rows, w, history)
	}
	scale := float32(1 / math.Sqrt(float64(a.width)))
	qWidth := a.query.out
	for start := 0; start < rows; start += min(a.block, rows-start) {
		end := min(rows, start+a.block)
		for row := start; row < end; row++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			for head := range a.heads {
				q := w.query[row*qWidth+head*a.width : row*qWidth+(head+1)*a.width]
				scores := w.scores[:end-start]
				for key := start; key < end; key++ {
					k := w.key[key*qWidth+head*a.width : key*qWidth+(head+1)*a.width]
					relative := a.relative[(row-key+a.radius)*a.width : (row-key+a.radius+1)*a.width]
					var dot, bias float32
					for d, v := range q {
						dot += v * k[d]
						bias += v * (relative[d] * scale)
					}
					scores[key-start] = dot*scale + bias
				}
				hostmath.SoftmaxInPlace(scores)
				out := w.attended[row*qWidth+head*a.width : row*qWidth+(head+1)*a.width]
				clear(out)
				for key, p := range scores {
					v := w.value[(key+start)*qWidth+head*a.width : (key+start)*qWidth+(head+1)*a.width]
					for d := range out {
						out[d] += p * v[d]
					}
				}
			}
		}
	}
	a.out.forward(w.branch, w.attended, rows)
	return ctx.Err()
}

func (c convolution) forward(ctx context.Context, x []float32, rows int, epsilon, layerEpsilon float64, w *Workspace, history []float32) error {
	c.in.forward(w.expanded, x, rows)
	// Compact GLU output within the expanded buffer, preserving unread rows.
	for row := range rows {
		for ch := range c.channels {
			gate := w.expanded[row*c.in.out+c.channels+ch]
			w.expanded[row*c.channels+ch] = w.expanded[row*c.in.out+ch] * float32(1/(1+math.Exp(-float64(gate))))
		}
	}
	if history != nil {
		n := rows * c.channels
		copy(w.expanded[len(history):len(history)+n], w.expanded[:n])
		copy(w.expanded, history)
		copy(history, w.expanded[n:n+len(history)])
	}
	// The residual consumes only floor(rows/stride), so the trailing convolution
	// row of an odd-length input need not be materialized.
	outRows := rows / c.stride
	for row := range outRows {
		if err := ctx.Err(); err != nil {
			return err
		}
		for ch := range c.channels {
			var sum float32
			for k := range c.kernelSize {
				position := row*c.stride + k - c.kernelSize/2
				if c.causal {
					position = row + k - (c.kernelSize - 1)
				}
				limit := rows
				if history != nil {
					position += c.kernelSize - 1
					limit += c.kernelSize - 1
				}
				if position >= 0 && position < limit {
					sum += w.expanded[position*c.channels+ch] * c.kernel[ch*c.kernelSize+k]
				}
			}
			if len(c.kernelBias) != 0 {
				sum += c.kernelBias[ch]
			}
			if len(c.layerNorm.weight) != 0 {
				w.conv[row*c.channels+ch] = sum
			} else {
				normalized := (float64(sum)-float64(c.mean[ch]))/math.Sqrt(float64(c.variance[ch])+epsilon)*float64(c.scale[ch]) + float64(c.bias[ch])
				w.conv[row*c.channels+ch] = float32(normalized)
			}
		}
	}
	if len(c.layerNorm.weight) != 0 {
		hostmath.LayerNormInto(w.conv, w.conv, c.layerNorm.weight, c.layerNorm.bias, outRows, c.channels, layerEpsilon)
	}
	hostmath.SiLUInPlace(w.conv[:outRows*c.channels])
	c.out.forward(w.branch, w.conv, outRows)
	return ctx.Err()
}

func (a attention) projectedRelative(ctx context.Context, rows int, w *Workspace, history *attentionHistory) error {
	keys, cached := rows, 0
	if history != nil {
		cached = history.frames
		keys += cached
		for _, pair := range []struct{ buffer, old []float32 }{{w.key, history.key}, {w.value, history.value}} {
			copy(pair.buffer[cached*a.query.out:keys*a.query.out], pair.buffer[:rows*a.query.out])
			copy(pair.buffer[:cached*a.query.out], pair.old[:cached*a.query.out])
		}
	}
	positions, width := 2*keys-1, a.position.in
	for p := range positions {
		for d := 0; d < width; d += 2 {
			frequency := float32(1 / math.Pow(a.positionBase, float64(d)/float64(width)))
			angle := float64(float32(keys-1-p) * frequency)
			w.relativeInput[p*width+d] = float32(math.Sin(angle))
			w.relativeInput[p*width+d+1] = float32(math.Cos(angle))
		}
	}
	a.position.forward(w.relativeKey, w.relativeInput, positions)
	scale, qWidth := float32(1/math.Sqrt(float64(a.width))), a.query.out
	leftChunks := a.leftContext / a.block
	for row := range rows {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := (cached + row) / a.block
		start, end := max(0, chunk-leftChunks)*a.block, min(keys, (chunk+1)*a.block)
		for head := range a.heads {
			scores := w.scores[:end-start]
			for key := start; key < end; key++ {
				var content, position float32
				for d := range a.width {
					channel := head*a.width + d
					q := w.query[row*qWidth+channel]
					content += (q + a.contentBias[channel]) * w.key[key*qWidth+channel]
					position += (q + a.positionBias[channel]) * w.relativeKey[(rows-1-row+key)*qWidth+channel]
				}
				scores[key-start] = content*scale + position*scale
			}
			hostmath.SoftmaxInPlace(scores)
			out := w.attended[row*qWidth+head*a.width : row*qWidth+(head+1)*a.width]
			clear(out)
			for key, probability := range scores {
				for d := range out {
					out[d] += probability * w.value[(start+key)*qWidth+head*a.width+d]
				}
			}
		}
	}
	a.out.forward(w.branch, w.attended, rows)
	if history != nil {
		history.frames = min(keys, a.leftContext)
		copy(history.key, w.key[(keys-history.frames)*qWidth:keys*qWidth])
		copy(history.value, w.value[(keys-history.frames)*qWidth:keys*qWidth])
	}
	return ctx.Err()
}
