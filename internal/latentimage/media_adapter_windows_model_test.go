//go:build modeltest && windows

package latentimage

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// unmergedAdapterTolerance is the unmerged adapter's bound against its merged weight, declared before its first run
// (media-lora/unmerged): over a joint block computed in F32 both ways, the unmerged output's largest difference from the
// merged one is at most 1e-4 of the merged output's largest magnitude, the reordered F32 sums of y = xW + s(xA)B against
// x(W + sAB) and nothing more.
const unmergedAdapterTolerance = 1e-4

// TestAnUnmergedAdapterMatchesItsMergedWeight holds an adapter applied unmerged (media-lora/unmerged) to the same
// adapter merged into its weights at F32: MOSS's first joint block, its weights MOSS-Image-SFT's, with the MOSS Ultra
// workflow's eight-step DMD2 adapter (kohya, rank 128, F32, on q, k, v, o and both feed-forward projections), over 64
// rows, both computed in F32. At strengths 1 and 0.5, fed to one compiled program, the unmerged output is within the
// declared tolerance of the merged one, and differs from the block without the adapter by more than the tolerance, so a
// program that dropped the adapter would fail it; and a second program compiled with the adapter binds its tensors from
// those already resident, uploading nothing.
func TestAnUnmergedAdapterMatchesItsMergedWeight(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	moss := testutil.ModelArtifactDir(t, "MOSS-Image-SFT-797fa8e678b4f")
	transformer := filepath.Join(moss, "transformer")
	spec := testcheck.Must(ReadRotaryTransformer(moss))(t)
	adapter := testcheck.Must(readMediaAdapter(filepath.Join(testutil.ModelArtifactDir(t, "moss-ultra"), "adapters", "8-steps-dmd2-lora.safetensors")))(t)
	const tokens, block = 64, 0
	session := testcheck.Must(graphruntime.NewResidentSession(0))(t)
	t.Cleanup(func() { testcheck.NoError(t, session.Close(context.WithoutCancel(ctx))) })

	draws := rand.New(rand.NewPCG(7, 11))
	random := func(count int, scale float64) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(draws.NormFloat64() * scale)
		}
		return values
	}
	hidden, modulation := random(spec.Hidden*tokens, 1), random(jointModulationFields*spec.Hidden, 0.1)
	rope := random(spec.HeadDim*tokens, 0.5)
	unit := make([]float32, spec.Hidden)
	for index := range unit {
		unit[index] = tensor.UnitScale
	}
	run := func(program *graphruntime.ResidentProgram, p *rotaryJointProgram, strength float32) []float32 {
		t.Helper()
		feeds := map[*tensor.Tensor]reference.Value{
			p.input: {Shape: p.input.Shape, Data: hidden}, p.modulation: {Shape: p.modulation.Shape, Data: modulation},
			p.unit: {Shape: p.unit.Shape, Data: unit}, p.rope: {Shape: p.rope.Shape, Data: rope},
		}
		for _, node := range p.strengths {
			feeds[node] = reference.Value{Shape: node.Shape, Data: []float32{strength}}
		}
		return testcheck.Must(session.Execute(ctx, program, feeds))(t)[p.output].Data
	}

	unmerged := testcheck.Must(compileRotaryJointProgram(spec, mossRefinementEpsilon, tokens, block, dtype.F32, rotaryOptions{adapters: []*mediaAdapter{adapter}}))(t)
	program := testcheck.Must(compileResidentWeights(ctx, session, "unmerged", transformer, unmerged.weightInputs, nil, unmerged.output))(t)
	testcheck.NoError(t, bindAdapterInputs(ctx, session, program, "unmerged", adapter, unmerged.adapterInputs[0]))
	resident := session.StaticBytes()
	again := testcheck.Must(compileRotaryJointProgram(spec, mossRefinementEpsilon, tokens, block, dtype.F32, rotaryOptions{adapters: []*mediaAdapter{adapter}}))(t)
	second := testcheck.Must(compileResidentWeights(ctx, session, "unmerged again", transformer, again.weightInputs, nil, again.output))(t)
	testcheck.NoError(t, bindAdapterInputs(ctx, session, second, "unmerged again", adapter, again.adapterInputs[0]))
	if uploaded := session.StaticBytes() - resident; uploaded != 0 {
		t.Errorf("a second program with the adapter uploaded %d bytes; its weights and the adapter's were resident", uploaded)
	}

	plain := testcheck.Must(compileRotaryJointProgram(spec, mossRefinementEpsilon, tokens, block, dtype.F32, rotaryOptions{}))(t)
	base := run(testcheck.Must(compileResidentWeights(ctx, session, "plain", transformer, plain.weightInputs, nil, plain.output))(t), plain, 0)
	deltas := adapterDeltas(t, adapter, fmt.Sprintf("layers.%d.", block))
	for _, strength := range []float32{1, 0.5} {
		got := run(program, unmerged, strength)
		merged := testcheck.Must(compileRotaryJointProgram(spec, mossRefinementEpsilon, tokens, block, dtype.F32, rotaryOptions{}))(t)
		mergedProgram := testcheck.Must(session.Compile(ctx, fmt.Sprintf("merged at %g", strength), nil, merged.output))(t)
		source := testcheck.Must(safetensors.OpenSource(transformer))(t)
		testcheck.NoError(t, bindResident(ctx, session, mergedProgram, "merged", fmt.Sprintf("merged at %g", strength), source, merged.weightInputs,
			func(source *safetensors.Source, node *tensor.Tensor) (func() ([]byte, error), error) {
				payload, err := residentPayload(source, node)
				delta, patched := deltas[strings.TrimSuffix(node.Name, ".weight")]
				if err != nil || !patched {
					return payload, err
				}
				return func() ([]byte, error) {
					raw, err := payload()
					for index := range delta {
						weight := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*index:]))
						binary.LittleEndian.PutUint32(raw[4*index:], math.Float32bits(weight+strength*delta[index]))
					}
					return raw, err
				}, nil
			}))
		testcheck.NoError(t, source.Close())
		want := run(mergedProgram, merged, strength)
		worst, magnitude, moved := 0.0, 0.0, 0.0
		for index, value := range want {
			worst = max(worst, math.Abs(float64(got[index]-value)))
			magnitude = max(magnitude, math.Abs(float64(value)))
			moved = max(moved, math.Abs(float64(value-base[index])))
		}
		t.Logf("at strength %g: unmerged against merged %.3g of the output's largest %.4g; the adapter moves it by up to %.4g", strength, worst/magnitude, magnitude, moved)
		if worst > unmergedAdapterTolerance*magnitude {
			t.Errorf("at strength %g the unmerged adapter departs from its merged weight by %.3g of the output's largest magnitude; the tolerance is %g",
				strength, worst/magnitude, unmergedAdapterTolerance)
		}
		if moved <= unmergedAdapterTolerance*magnitude {
			t.Errorf("at strength %g the adapter moves the block's output by %.4g, within the tolerance, so a program that dropped it would pass", strength, moved)
		}
	}
}

// adapterDeltas are the adapter's merged differences, scale * up * down in F32, of the projections of the block whose
// names open with prefix, by projection, each laid out as the weight's payload, a row per output.
func adapterDeltas(t *testing.T, adapter *mediaAdapter, prefix string) map[string][]float32 {
	t.Helper()
	source := testcheck.Must(safetensors.OpenFile(adapter.path))(t)
	defer source.Close()
	deltas := map[string][]float32{}
	for name, patch := range adapter.lowRank {
		if !strings.HasPrefix(name, prefix) || patch.split {
			continue
		}
		down := testcheck.Must(safetensors.ReadF32(source.Tensors[patch.down]))(t)
		up := testcheck.Must(safetensors.ReadF32(source.Tensors[patch.up]))(t)
		in, out, rank := int(patch.in), int(patch.out), int(patch.rank)
		delta := make([]float32, out*in)
		var group sync.WaitGroup
		rows := make(chan int, out)
		for row := range out {
			rows <- row
		}
		close(rows)
		for range runtime.NumCPU() {
			group.Go(func() {
				for row := range rows {
					for column := range in {
						sum := 0.0
						for r := range rank {
							sum += float64(up[row*rank+r]) * float64(down[r*in+column])
						}
						delta[row*in+column] = float32(sum) * patch.scale
					}
				}
			})
		}
		group.Wait()
		deltas[name] = delta
	}
	return deltas
}
