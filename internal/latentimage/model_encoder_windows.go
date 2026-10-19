//go:build windows

package latentimage

import (
	"context"
	"errors"
	"io"
	"overgo/internal/graphruntime"
	"overgo/internal/hfgguf"
	"overgo/internal/tensor"
)

func (p *modelEncoderProgram) resident(ctx context.Context, session *graphruntime.ResidentSession, catalog hfgguf.LanguageCatalog, sourceIdentity string, outputs ...*tensor.Tensor) (_ *graphruntime.ResidentProgram, err error) {
	if len(outputs) == 0 {
		outputs = []*tensor.Tensor{p.output}
	}
	program, err := session.Compile(ctx, "model conditioning", nil, outputs...)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Release(context.WithoutCancel(ctx), program))
		}
	}()
	for _, node := range p.weights {
		if _, required := program.Graph.InputSlot(node); !required {
			continue
		}
		if err = session.Bind(ctx, program, sourceIdentity, node.Name, node, func() ([]byte, error) {
			payload, err := catalog.Payload(node.Name)
			if err != nil {
				return nil, err
			}
			return io.ReadAll(payload.Data)
		}); err != nil {
			return nil, err
		}
	}
	return program, nil
}
