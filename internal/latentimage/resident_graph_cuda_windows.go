//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
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

// bindMerged binds each projection a compiled program reads with adapters' pairs merged into its int8 weight
// (mergedProjection) as the merge on the card makes it (convRotMerge), each adapter at its strength, from the int8 weight
// the session keeps on the card as the transformer's file at sourcePath holds it (ResidentSession.Keep), and from each
// adapter's pairs the session keeps on the card under the adapter's content and the tensor's name, read from its file
// the first time alone, whatever sections since merged other adapters (moss-merge-compute): read, rounded and uploaded
// again for each request, they had cost text to image Turbo 1.8 s of its sampling. A merge is held under the
// transformer's path, each merged adapter's content and strength, and the weight's name, so a later sampling with those
// adapters at those strengths takes it as it is, and one whose adapter file was replaced under its name merges it again
// (moss-merge-time/identity): held under the adapter's path, it answered a replaced file with the earlier file's merge
// (SQA-406's fault in the merge's place).
func bindMerged(ctx context.Context, session *graphruntime.ResidentSession, program *graphruntime.ResidentProgram, label, sourcePath string,
	merged []mergedProjection, adapters []*mediaAdapter, strengths []float32) (err error) {
	if len(merged) == 0 {
		return nil
	}
	source, err := safetensors.OpenSource(sourcePath)
	if err != nil {
		return fmt.Errorf("%s: open weights: %w", label, err)
	}
	defer func() { err = errors.Join(err, source.Close()) }()
	place := func(adapter *mediaAdapter, name string, rows, columns uint64) (driver.DevicePtr, error) {
		node := &tensor.Tensor{Name: name, Type: dtype.BF16, Shape: tensor.MustShape(columns, rows)}
		return session.Keep(ctx, adapter.content, name, node, func() ([]byte, error) { return adapter.bf16Tensor(name) })
	}
	for _, projection := range merged {
		if _, required := program.Graph.InputSlot(projection.input); !required {
			continue
		}
		pairs := make([]adapterPair, len(projection.pairs))
		var identity strings.Builder
		identity.WriteString(sourcePath)
		for index, pair := range projection.pairs {
			if pair.adapter >= len(adapters) || pair.adapter >= len(strengths) {
				return fmt.Errorf("%s: %s merges adapter %d of %d", label, projection.input.Name, pair.adapter, len(adapters))
			}
			pairs[index] = adapterPair{adapter: adapters[pair.adapter], patch: pair.patch, strength: strengths[pair.adapter]}
			identity.WriteString(fmt.Sprintf("+%s@%g", adapters[pair.adapter].content, strengths[pair.adapter]))
		}
		name := projection.input.Name
		if err := session.BindMerged(ctx, program, identity.String(), name, projection.input, func() (executor.Int8ConvRotMerge, error) {
			// The weight it merges into is the one the session keeps on the card as the file holds it, bound by an
			// earlier program or read from the file the first time alone, whatever the programs since bound merged
			// (moss-merge-order).
			payload, err := residentPayload(source, projection.input)
			if err != nil {
				return executor.Int8ConvRotMerge{}, err
			}
			held, err := session.Keep(ctx, sourcePath, name, projection.input, payload)
			if err != nil {
				return executor.Int8ConvRotMerge{}, err
			}
			return convRotMerge(source, name, pairs, place, held)
		}); err != nil {
			return fmt.Errorf("%s: %s: %w", label, name, err)
		}
	}
	return nil
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
