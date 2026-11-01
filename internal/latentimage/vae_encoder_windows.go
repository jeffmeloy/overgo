//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// PictureEncoder encodes pictures of one extent to the rows a rotary
// transformer reads, on a device session of its own.
type PictureEncoder struct {
	session *graphruntime.ResidentSession
	encoder *VAEEncoder
	graph   *VAEEncoderProgram
	program *graphruntime.ResidentProgram
}

// NewPictureEncoder places a checkpoint's encoder on the device for pictures of
// height by width.
func NewPictureEncoder(ctx context.Context, root string, height, width int) (_ *PictureEncoder, err error) {
	encoder, err := loadSpatialVAEEncoder(root)
	if err != nil {
		return nil, err
	}
	graph, err := compileVAEEncoderProgram(encoder, height, width)
	if err != nil {
		return nil, err
	}
	session, err := graphruntime.NewResidentSession(tensor.FirstOffset)
	if err != nil {
		return nil, fmt.Errorf("picture encoder: device: %w", err)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Close(context.WithoutCancel(ctx)))
		}
	}()
	static := make(map[*tensor.Tensor][]float32, len(graph.feeds))
	for _, feed := range graph.feeds {
		static[feed.node] = feed.data
	}
	program, err := compileResidentStatic(ctx, session, "picture encoder", static, nil, graph.Mean)
	if err != nil {
		return nil, err
	}
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

// Close releases the encoder's device state.
func (e *PictureEncoder) Close(ctx context.Context) error {
	if e == nil || e.session == nil {
		return nil
	}
	session := e.session
	e.session = nil
	return session.Close(context.WithoutCancel(ctx))
}
