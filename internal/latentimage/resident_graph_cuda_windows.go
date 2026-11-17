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
	return program, bindResident(ctx, session, program, label, sourcePath, source, inputs, residentPayload)
}

// bindAdapterInputs binds an adapter's tensors a compiled program reads from its file (adapterPayload), each uploaded
// once for every program that reads it.
func bindAdapterInputs(ctx context.Context, session *graphruntime.ResidentSession, program *graphruntime.ResidentProgram, label string,
	adapter *mediaAdapter, inputs tensor.WeightBindings) error {
	source, err := safetensors.OpenFile(adapter.path)
	if err != nil {
		return fmt.Errorf("%s: open adapter: %w", label, err)
	}
	defer source.Close()
	return bindResident(ctx, session, program, label, adapter.path, source, inputs, adapterPayload)
}

// bindResident binds each input the program reads from source by payload, keyed by the source's path and the input's
// name.
func bindResident(ctx context.Context, session *graphruntime.ResidentSession, program *graphruntime.ResidentProgram, label, sourcePath string,
	source *safetensors.Source, inputs tensor.WeightBindings, payloadOf func(*safetensors.Source, *tensor.Tensor) (func() ([]byte, error), error)) error {
	for _, node := range inputs {
		if _, required := program.Graph.InputSlot(node); !required {
			continue
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		payload, err := payloadOf(source, node)
		if err == nil {
			err = session.Bind(ctx, program, sourcePath, node.Name, node, payload)
		}
		if err != nil {
			return fmt.Errorf("%s: tensor %s: %w", label, node.Name, err)
		}
	}
	return nil
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
