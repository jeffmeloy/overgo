//go:build windows

package latentimage

import (
	"context"
	"fmt"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
	"path/filepath"
	"testing"
)

func TestMOSSRotaryJointCUDAGolden(t *testing.T) {
	cudatest.Require(t)
	oracle := jointFixture(t)
	spec := refinementSpec()
	spec.Layers = 2
	runtime := testcheck.Must(graphruntime.NewResidentSession(0))(t)
	defer runtime.Close(context.WithoutCancel(t.Context()))
	for _, length := range []int{2, 4} {
		for block := range 2 {
			t.Run(fmt.Sprintf("%d/%d", length, block), func(t *testing.T) {
				p := testcheck.Must(compileRotaryJointProgram(spec, 1e-6, length+6, block, dtype.BF16))(t)
				prefix := fmt.Sprintf("cuda-bf16.%d.block.%d.", length, block)
				program := testcheck.Must(runtime.Compile(t.Context(), "joint block", nil, p.output))(t)
				defer runtime.Release(context.WithoutCancel(t.Context()), program)
				feeds := jointFeeds(t, p, oracle, prefix, dtype.BF16)
				for _, node := range p.weightInputs {
					values := feeds[node].Data
					err := runtime.Bind(t.Context(), program, "joint fixture", node.Name, node, storedWeight(node, values))
					testcheck.NoError(t, err)
					delete(feeds, node)
				}
				got := testcheck.Must(runtime.Execute(t.Context(), program, feeds))(t)
				compareRefinement(t, got[p.output].Data, refinementValues(t, oracle, prefix+"output"), .016)
				repeat := testcheck.Must(runtime.Execute(t.Context(), program, feeds))(t)
				for i, x := range got[p.output].Data {
					if repeat[p.output].Data[i] != x {
						t.Fatal("joint block did not repeat exactly")
					}
				}
			})
		}
	}
}

func TestMOSSRotaryJointRMSBoundary(t *testing.T) {
	cudatest.Require(t)
	mossFixtureBytes(t, "joint-rms.safetensors", "8dee4bd3e3f65dfc431e974444f2fa081e176af8930602c6c08c4b48b2bf48ce")
	source := testcheck.Must(safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "joint-rms.safetensors")))(t)
	defer source.Close()
	runtime := testcheck.Must(graphruntime.NewResidentSession(0))(t)
	defer runtime.Close(context.WithoutCancel(t.Context()))
	b := tensor.NewBuilder()
	x := b.Input("x", dtype.F32, tensor.MustShape(3072))
	y := b.BF16Round(b.RMSNorm(x, 1e-6))
	program := testcheck.Must(runtime.Compile(t.Context(), "native joint RMS", nil, y))(t)
	defer runtime.Release(context.WithoutCancel(t.Context()), program)
	got := testcheck.Must(runtime.Execute(t.Context(), program, map[*tensor.Tensor]reference.Value{x: {Shape: x.Shape, Data: refinementValues(t, source, "input")}}))(t)
	want := refinementValues(t, source, "rounded")
	for i, value := range got[y].Data {
		if value != want[i] {
			t.Errorf("native RMS boundary %d got=%g want=%g", i, value, want[i])
		}
	}
}
