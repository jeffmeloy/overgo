package densecausal

import (
	"context"
	"errors"

	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// blockProgram names the adapter's programs and what they hold on a device.
const blockProgram = "densecausal block"

// onDevice moves the adapter's programs to a device: the same two graphs are
// compiled for the device executor, and the model's weights and the backward
// graph's constants are uploaded once and held there, so a step sends the
// state, the target and the factors and reads back the miss and the factors'
// gradients. The factors and Muon's state stay the trainer's own values on
// the host. The session returned is what the device holds: closing it
// releases that, and the adapter is not used after.
func (adapter *blockAdapter) onDevice(ctx context.Context, ordinal int) (*graphruntime.ResidentSession, error) {
	session, err := graphruntime.NewResidentSession(ordinal)
	if err != nil {
		return nil, err
	}
	release := func() error { return session.Close(ctx) }
	held := func(outputs []*tensor.Tensor) (*graphruntime.ResidentProgram, error) {
		program, err := session.Compile(ctx, blockProgram, nil, outputs...)
		if err != nil {
			return nil, err
		}
		for _, kept := range []map[*tensor.Tensor][]float32{adapter.fixed, adapter.constants} {
			for node, values := range kept {
				if _, read := program.Graph.InputSlot(node); !read {
					continue
				}
				if err := session.BindF32(ctx, program, blockProgram, node, values); err != nil {
					return nil, err
				}
			}
		}
		return program, nil
	}
	forward, err := held(adapter.outputs[:1])
	if err != nil {
		return nil, errors.Join(err, release())
	}
	both, err := held(adapter.outputs)
	if err != nil {
		return nil, errors.Join(err, release())
	}
	clear(adapter.fixed)
	clear(adapter.constants)
	adapter.execute = func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		program := both
		if len(outputs) == 1 {
			program = forward
		}
		return session.Execute(ctx, program, feeds)
	}
	return session, nil
}
