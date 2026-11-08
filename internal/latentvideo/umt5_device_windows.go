//go:build windows

package latentvideo

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"overgo/internal/binaryschema"
	"overgo/internal/graphruntime"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// umt5Device is the text encoder resident on the card: its weights bound once into one session by name, and a program
// for each sequence length compiled when a prompt of it first comes, which takes the weights the session holds. The
// host encoder streams the 11 GB checkpoint through the CPU for every new prompt.
type umt5Device struct {
	session    *graphruntime.ResidentSession
	checkpoint string
	plan       encoderPlan
	byName     map[string]pytorchzip.TensorBinding
	programs   map[int]umt5Resident
}

// umt5Resident is one sequence length's compiled program and the graph it was compiled from.
type umt5Resident struct {
	program *graphruntime.ResidentProgram
	graph   *umt5Program
}

func init() {
	openCardEncoder = func(checkpoint string, plan encoderPlan) (cardEncoder, error) {
		return openUMT5Device(checkpoint, plan, 0)
	}
}

// openUMT5Device opens the encoder of the checkpoint, under its validated plan, on the card of ordinal.
func openUMT5Device(checkpoint string, plan encoderPlan, ordinal int) (*umt5Device, error) {
	byName, err := encoderBindingsByName(plan)
	if err != nil {
		return nil, err
	}
	session, err := graphruntime.NewResidentSession(ordinal)
	if err != nil {
		return nil, err
	}
	return &umt5Device{session: session, checkpoint: checkpoint, plan: plan, byName: byName, programs: map[int]umt5Resident{}}, nil
}

// encode gives a sequence's final rows, a row of Dim values for each token, rounded to BF16 as the host encoder's are
// (encodeSequencesStreamed).
func (d *umt5Device) encode(ctx context.Context, tokens []int) ([]float32, error) {
	if d == nil || d.session == nil || len(tokens) == 0 {
		return nil, errors.New("umt5 device: no encoder or no tokens")
	}
	reader, err := pytorchzip.Open(d.checkpoint)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	resident, err := d.programFor(ctx, reader, len(tokens))
	if err != nil {
		return nil, err
	}
	embed, err := reader.ReadTensorRows(d.byName["token_embedding.weight"], tokens, d.plan.Config.Dim)
	if err != nil {
		return nil, err
	}
	outputs, err := d.session.Execute(ctx, resident.program, map[*tensor.Tensor]reference.Value{
		resident.graph.Embed: {Shape: resident.graph.Embed.Shape, Data: embed},
	})
	if err != nil {
		return nil, fmt.Errorf("umt5 device: %w", err)
	}
	rows, held := outputs[resident.graph.Output]
	if !held {
		return nil, errors.New("umt5 device: the encoder's rows are unavailable")
	}
	dtype.RoundBF16Slice(rows.Data)
	return rows.Data, nil
}

// programFor is the program of seq tokens: the one compiled before, or one compiled now, its weights bound from the
// checkpoint where the session holds none of that name yet.
func (d *umt5Device) programFor(ctx context.Context, reader *pytorchzip.Reader, seq int) (umt5Resident, error) {
	if resident, held := d.programs[seq]; held {
		return resident, nil
	}
	graph, err := compileUMT5Program(d.plan.Config, seq)
	if err != nil {
		return umt5Resident{}, err
	}
	program, err := d.session.Compile(ctx, "umt5 encoder", nil, graph.Output)
	if err != nil {
		return umt5Resident{}, err
	}
	for _, node := range graph.weights {
		binding, named := d.byName[node.Name]
		if !named {
			return umt5Resident{}, errors.Join(fmt.Errorf("umt5 device: the checkpoint binds no %s", node.Name), d.session.Release(context.WithoutCancel(ctx), program))
		}
		if err := d.session.Bind(ctx, program, d.checkpoint, node.Name, node, func() ([]byte, error) { return umt5Payload(reader, binding, node.Type) }); err != nil {
			return umt5Resident{}, errors.Join(fmt.Errorf("umt5 device: %s: %w", node.Name, err), d.session.Release(context.WithoutCancel(ctx), program))
		}
	}
	d.programs[seq] = umt5Resident{program: program, graph: graph}
	return d.programs[seq], nil
}

// umt5Payload is a tensor's bytes in the storage its graph input reads: a matrix's BF16 as stored, anything else as F32.
func umt5Payload(reader *pytorchzip.Reader, binding pytorchzip.TensorBinding, storage dtype.Type) ([]byte, error) {
	if storage == dtype.BF16 {
		values, err := reader.ReadBF16(binding, nil)
		data := make([]byte, len(values)*binaryschema.Uint16Bytes)
		for index, value := range values {
			binary.LittleEndian.PutUint16(data[index*binaryschema.Uint16Bytes:], value)
		}
		return data, err
	}
	values, err := reader.ReadBinding(binding)
	data := make([]byte, len(values)*binaryschema.Uint32Bytes)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*binaryschema.Uint32Bytes:], math.Float32bits(value))
	}
	return data, err
}

// Close gives the encoder's programs and weights back to the card.
func (d *umt5Device) Close(ctx context.Context) error {
	if d == nil || d.session == nil {
		return nil
	}
	err := d.session.Close(ctx)
	d.session, d.programs = nil, nil
	return err
}
