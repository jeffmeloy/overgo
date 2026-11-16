//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

func compileResidentWeights(
	ctx context.Context,
	session *graphruntime.ResidentSession,
	label, sourcePath string,
	inputs tensor.WeightBindings,
	dynamic []*tensor.Tensor,
	outputs ...*tensor.Tensor,
) (_ *graphruntime.ResidentProgram, err error) {
	program, err := session.Compile(ctx, label, dynamic, outputs...)
	if err != nil || len(inputs) == 0 {
		return program, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Release(context.WithoutCancel(ctx), program))
		}
	}()
	source, err := safetensors.OpenSource(sourcePath)
	if err != nil {
		return nil, fmt.Errorf("%s: open weights: %w", label, err)
	}
	defer source.Close()
	for _, node := range inputs {
		if _, required := program.Graph.InputSlot(node); !required {
			continue
		}
		name := node.Name
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		payload, err := residentPayload(source, node)
		if err == nil {
			err = session.Bind(ctx, program, sourcePath, name, node, payload)
		}
		if err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", label, name, err)
		}
	}
	return program, nil
}

func compileResidentStatic(
	ctx context.Context,
	session *graphruntime.ResidentSession,
	label string,
	inputs map[*tensor.Tensor][]float32,
	dynamic []*tensor.Tensor,
	outputs ...*tensor.Tensor,
) (*graphruntime.ResidentProgram, error) {
	program, err := session.Compile(ctx, label, dynamic, outputs...)
	if err != nil {
		return nil, err
	}
	for node, values := range inputs {
		if err := session.BindF32(ctx, program, "static:"+label, node, values); err != nil {
			return nil, errors.Join(fmt.Errorf("%s: tensor %s: %w", label, node.Name, err), session.Release(context.Background(), program))
		}
	}
	return program, nil
}
