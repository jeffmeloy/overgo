//go:build windows

package latentimage

import (
	"context"
	"errors"
	"path/filepath"

	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// CaptionConditioner gives a training run what the frozen text path makes of a
// caption: the text encoder's hidden rows through the transformer's text
// refinement, in the arithmetic inference serves. The first caption's programs
// hold the weights, so the captions after it place none.
type CaptionConditioner struct {
	session     *graphruntime.ResidentSession
	conditioner *modelConditioner
	spec        TransformerSpec
	root        string
	held        []*graphruntime.ResidentProgram
}

// NewCaptionConditioner opens the checkpoint's text path on a device session of its own.
func NewCaptionConditioner(root string, profile Profile) (_ *CaptionConditioner, err error) {
	actual, err := ResolveProfile(root)
	if err != nil {
		return nil, err
	}
	if actual.ID != profile.ID {
		return nil, errors.New("caption conditioner: recipe profile differs from artifact")
	}
	spec, err := ReadRotaryTransformer(root)
	if err != nil {
		return nil, err
	}
	conditioner, err := loadModelConditioner(root, profile.Prompt)
	if err != nil {
		return nil, err
	}
	session, err := graphruntime.NewResidentSession(tensor.FirstOffset)
	if err != nil {
		return nil, errors.Join(err, conditioner.source.Close())
	}
	return &CaptionConditioner{session: session, conditioner: conditioner, spec: spec, root: root}, nil
}

// Refine returns a caption's refined rows [tokens, hidden width] and its token count.
func (c *CaptionConditioner) Refine(ctx context.Context, caption string) (_ []float32, tokens int, err error) {
	text, embedding, err := c.conditioner.input(caption)
	if err != nil {
		return nil, 0, err
	}
	var anchor *[]*graphruntime.ResidentProgram
	if len(c.held) == 0 {
		anchor = &c.held
	}
	hidden, err := c.conditioner.encodeHolding(ctx, c.session, anchor, text, embedding, nil)
	if err != nil {
		return nil, 0, err
	}
	tokens = len(text.IDs)
	graph, err := compileRotaryRefinementProgram(c.spec, mossRefinementEpsilon, tokens, dtype.BF16)
	if err != nil {
		return nil, 0, err
	}
	program, err := compileResidentWeights(ctx, c.session, "caption refinement", filepath.Join(c.root, "transformer"), graph.weightInputs, nil, graph.Fused)
	if err != nil {
		return nil, 0, err
	}
	if len(c.held) == 1 {
		c.held = append(c.held, program)
	} else {
		defer func() { err = errors.Join(err, c.session.Release(context.WithoutCancel(ctx), program)) }()
	}
	// A token of text has no picture coordinates, so the table of any picture holds its row.
	rope, err := CoordinateRotary(c.spec.HeadDim, tensor.SingletonExtent, tensor.SingletonExtent, tokens, c.spec.RopeTheta)
	if err != nil {
		return nil, 0, err
	}
	result, err := c.session.Execute(ctx, program, map[*tensor.Tensor]reference.Value{
		graph.InEncoder: {Shape: graph.InEncoder.Shape, Data: hidden},
		graph.InRoPE:    {Shape: graph.InRoPE.Shape, Data: rope[c.spec.HeadDim:]},
	})
	if err != nil {
		return nil, 0, err
	}
	return result[graph.Fused].Data, tokens, nil
}

// Close releases the text path's device state.
func (c *CaptionConditioner) Close(ctx context.Context) error {
	if c == nil || c.session == nil {
		return nil
	}
	session := c.session
	c.session = nil
	return errors.Join(session.Close(context.WithoutCancel(ctx)), c.conditioner.source.Close())
}
