//go:build windows

package latentimage

import (
	"context"
	"fmt"
	"testing"

	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

type residentFixture struct {
	runtime *graphruntime.ResidentSession
	graph   *graphruntime.ResidentProgram
}

func newResidentFixture(
	t *testing.T,
	ctx context.Context,
	label, source string,
	inputs map[string]*tensor.Tensor,
	outputs ...*tensor.Tensor,
) *residentFixture {
	t.Helper()
	runtime, err := graphruntime.NewResidentSession(0)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := compileResidentWeights(ctx, runtime, label, source, inputs, nil, outputs...)
	if err != nil {
		_ = runtime.Close(ctx)
		t.Fatal(err)
	}
	fixture := &residentFixture{runtime: runtime, graph: graph}
	t.Cleanup(func() {
		if err := fixture.runtime.Close(ctx); err != nil {
			t.Errorf("close resident fixture: %v", err)
		}
	})
	return fixture
}

func (f *residentFixture) execute(
	t *testing.T,
	ctx context.Context,
	feeds map[*tensor.Tensor]reference.Value,
) map[*tensor.Tensor]reference.Value {
	t.Helper()
	results, err := f.runtime.Execute(ctx, f.graph, feeds)
	if err != nil {
		t.Fatal(err)
	}
	return results
}

func residentEncode(t *testing.T, ctx context.Context, fixture *residentFixture, program *EncoderProgram, embed []float32) *SelectedHiddenStates {
	t.Helper()
	feeds := map[*tensor.Tensor]reference.Value{program.Embed: {Shape: program.Embed.Shape, Data: embed}}
	if program.keyBias != nil {
		feeds[program.keyBias] = reference.Value{Shape: program.keyBias.Shape, Data: program.keyBiasData}
	}
	result, err := program.assemble(fixture.execute(t, ctx, feeds))
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func residentFuse(t *testing.T, ctx context.Context, fixture *residentFixture, program *FusionProgram, hidden []float32) []float32 {
	t.Helper()
	result, err := program.assemble(fixture.execute(t, ctx, map[*tensor.Tensor]reference.Value{
		program.InEncoder: {Shape: program.InEncoder.Shape, Data: hidden},
	}))
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func residentDenoise(
	t *testing.T,
	ctx context.Context,
	fixture *residentFixture,
	program *DenoiserProgram,
	latent, text, embedding, modulation []float32,
) ForwardResult {
	t.Helper()
	results := fixture.execute(t, ctx, map[*tensor.Tensor]reference.Value{
		program.InLatent:  {Shape: program.InLatent.Shape, Data: latent},
		program.InText:    {Shape: program.InText.Shape, Data: text},
		program.InTemb:    {Shape: program.InTemb.Shape, Data: embedding},
		program.InTembMod: {Shape: program.InTembMod.Shape, Data: modulation},
		program.InDelta:   {Shape: program.InDelta.Shape, Data: []float32{0}},
	})
	full := results[program.Velocity].Data
	if len(full) != program.Seq*program.T.InChannels {
		t.Fatal(fmt.Errorf("resident fixture: velocity len=%d", len(full)))
	}
	result := ForwardResult{
		Velocity:    append([]float32(nil), full[program.TextSeq*program.T.InChannels:]...),
		BlockHidden: make([][]float32, len(program.BlockOutputs)),
	}
	for index, output := range program.BlockOutputs {
		result.BlockHidden[index] = results[output].Data
	}
	return result
}
