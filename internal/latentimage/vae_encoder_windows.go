//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// PictureEncoder encodes pictures of one extent to the rows a rotary
// transformer reads, on a device session of its own or one another holds. It
// keeps its bound program and what the latent's normalization reads, not the
// host copies of the weights it put on the card (SQA-435).
type PictureEncoder struct {
	session *graphruntime.ResidentSession
	// owned is whether the encoder opened its session, and closes it.
	owned   bool
	encoder *VAEEncoder
	graph   *VAEEncoderProgram
	program *graphruntime.ResidentProgram
}

// NewPictureEncoder places a checkpoint's encoder on the device for pictures of
// height by width, in a session of its own.
func NewPictureEncoder(ctx context.Context, root string, height, width int) (_ *PictureEncoder, err error) {
	session, err := graphruntime.NewResidentSession(tensor.FirstOffset)
	if err != nil {
		return nil, fmt.Errorf("picture encoder: device: %w", err)
	}
	encoder, err := newPictureEncoderIn(ctx, session, root, height, width)
	if err != nil {
		return nil, errors.Join(err, session.Close(context.WithoutCancel(ctx)))
	}
	encoder.owned = true
	return encoder, nil
}

// newPictureEncoderIn places a checkpoint's encoder for pictures of height by
// width in a session another holds, which its Close leaves open.
func newPictureEncoderIn(ctx context.Context, session *graphruntime.ResidentSession, root string, height, width int) (*PictureEncoder, error) {
	encoder, err := loadSpatialVAEEncoder(root)
	if err != nil {
		return nil, err
	}
	graph, err := compileVAEEncoderProgram(encoder, height, width)
	if err != nil {
		return nil, err
	}
	static := make(map[*tensor.Tensor][]float32, len(graph.feeds))
	for _, feed := range graph.feeds {
		static[feed.node] = feed.data
	}
	program, err := compileResidentStatic(ctx, session, "picture encoder", static, nil, graph.Mean)
	if err != nil {
		return nil, err
	}
	// The weights are on the card; Encode reads the bound program, the geometry and the latent's statistics alone.
	graph.feeds, graph.boundaries, encoder.CodecProgram = nil, nil, media.CodecProgram[[]float32]{}
	return &PictureEncoder{session: session, encoder: encoder, graph: graph, program: program}, nil
}

// mean runs the encoder over [channels, height*width] rows in the signed unit
// range and returns the mean half of its moments, [latent channels, h*w] rows.
func (e *PictureEncoder) mean(ctx context.Context, pixels []float32) ([]float32, error) {
	if e == nil || e.session == nil {
		return nil, errors.New("picture encoder: unavailable")
	}
	result, err := e.session.Execute(ctx, e.program, map[*tensor.Tensor]reference.Value{e.graph.Pixels: {Shape: e.graph.Pixels.Shape, Data: pixels}})
	if err != nil {
		return nil, err
	}
	return result[e.graph.Mean].Data, nil
}

// Encode returns a picture's latent rows [grid height*grid width, channels],
// patched and normalized as the transformer reads them, with the grid.
func (e *PictureEncoder) Encode(ctx context.Context, pixels []float32) (rows []float32, gridHeight, gridWidth int, err error) {
	mean, err := e.mean(ctx, pixels)
	if err != nil {
		return nil, 0, 0, err
	}
	return e.encoder.normalizedLatent(mean, e.graph.H, e.graph.W)
}

// Close releases the encoder's program and its weights, and the session where the encoder opened it.
func (e *PictureEncoder) Close(ctx context.Context) error {
	if e == nil || e.session == nil {
		return nil
	}
	session := e.session
	e.session = nil
	if e.owned {
		return session.Close(context.WithoutCancel(ctx))
	}
	return session.Release(context.WithoutCancel(ctx), e.program)
}
