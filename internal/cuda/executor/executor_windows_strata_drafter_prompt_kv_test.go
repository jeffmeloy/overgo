//go:build windows

package executor

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
)

// The batched pass's fixture: prompt cells past one batch, so a batch boundary and a short last batch are both read, a
// draft vocabulary of embedding rows, and the most the two passes' K or V rows may part, relative to the row: the
// six-row pass multiplies q8_1 activations, the batched pass f16 ones.
const (
	strataBatchedCells     = strataDraftBatchRows + 44
	strataBatchedVocabRows = 64
	strataBatchedTolerance = 0.03
)

// TestDrafterPromptKVBatchedAsItsRows holds the batched pass of the drafter's prompt K/V (Prefill::draft_kv, E-9) to
// the cells its six-row pass (E-4) writes from the same rows and tokens: every cell's K and V, each head's row, within
// the products' rounding, over a batch boundary and a short last batch. The weights are random in the drafter's
// formats: the fc, K and V projections and the token embedding in Q8_0, the hyper-connection's down and up in BF16.
func TestDrafterPromptKVBatchedAsItsRows(t *testing.T) {
	e := newFixtureExecutor(t)
	random := rand.New(rand.NewPCG(7, 11))
	n, hcn := uint64(strataGREmbedding), uint64(strataGRStreams*strataGREmbedding)
	heads, width, page := uint64(tensor.QSAKeyValueHeads), uint64(tensor.QSAHeadWidth), uint64(tensor.QSABlockCells)
	kv := heads * width
	normal := func(count int, scale float64) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(random.NormFloat64() * scale)
		}
		return values
	}
	gamma := func(count int) []float32 {
		values := normal(count, 0.1)
		for index := range values {
			values[index]++
		}
		return values
	}
	// q80 is a rows x n Q8_0 matrix of codes in [-64, 64] at scale, row after row.
	q80 := func(rows uint64, scale float32) []byte {
		data := make([]byte, 0, rows*(n/32*34))
		for range rows * n / 32 {
			data = binary.LittleEndian.AppendUint16(data, quant.Float32ToFloat16(scale))
			for range 32 {
				data = append(data, byte(int8(random.IntN(129)-64)))
			}
		}
		return data
	}
	bf16 := func(values []float32) []byte {
		data := make([]byte, 0, 2*len(values))
		for _, value := range values {
			bits := math.Float32bits(value)
			bits += 0x7fff + (bits>>16)&1
			data = binary.LittleEndian.AppendUint16(data, uint16(bits>>16))
		}
		return data
	}
	projection := float32(1 / (64 * math.Sqrt(float64(n))))
	uploads := []struct {
		target *driver.DevicePtr
		data   []byte
	}{}
	var spec StrataDrafterSpec
	var rows driver.DevicePtr
	spec.MaxTokens, spec.EmbeddingRow = strataPrefillMaxT, n/32*34
	for _, weight := range []*StrataRoundWeight{&spec.FCEmbedding, &spec.FCHidden} {
		weight.Type = strataGGMLQ8_0
		uploads = append(uploads, struct {
			target *driver.DevicePtr
			data   []byte
		}{&weight.Pointer, q80(n, projection)})
	}
	for _, weight := range []*StrataRoundWeight{&spec.Key, &spec.Value} {
		weight.Type = strataGGMLQ8_0
		uploads = append(uploads, struct {
			target *driver.DevicePtr
			data   []byte
		}{&weight.Pointer, q80(kv, projection)})
	}
	for _, upload := range []struct {
		target *driver.DevicePtr
		data   []byte
	}{
		{&spec.Embedding, q80(strataBatchedVocabRows, 1.0/64)},
		{&spec.EmbeddingNorm, float32Bytes(gamma(int(n)))},
		{&spec.HiddenNorm, float32Bytes(gamma(int(hcn)))},
		{&spec.KeyNorm, float32Bytes(gamma(int(width)))},
		{&spec.Attention.Norm, float32Bytes(gamma(int(hcn)))},
		{&spec.Attention.Down, bf16(normal(strataGRRank*int(hcn), 1/math.Sqrt(float64(hcn))))},
		{&spec.Attention.Up, bf16(normal(int(hcn)*strataGRRank, 1/math.Sqrt(strataGRRank)))},
		{&rows, float32Bytes(normal(strataBatchedCells*int(hcn), 1))},
	} {
		uploads = append(uploads, upload)
	}
	pages := int64((strataBatchedCells + page - 1) / page)
	drafter := func() *StrataDrafter {
		return &StrataDrafter{spec: spec, resources: &e.resources, pages: pages, slots: pages,
			rope: strataRopeScaling{freqBase: 1e7, factor: 1, attnFactor: 1}}
	}
	var groups, batched *StrataDrafter
	if err := e.worker.Do(t.Context(), func(state *device.State) error {
		for _, upload := range uploads {
			pointer, err := e.resources.allocate(state, uint64(len(upload.data)))
			if err != nil {
				return err
			}
			*upload.target = pointer
			if err := state.Driver.MemcpyHtoD(pointer, upload.data); err != nil {
				return err
			}
		}
		groups, batched = drafter(), drafter()
		groups.spec, batched.spec = spec, spec
		for _, d := range []*StrataDrafter{groups, batched} {
			if err := d.loadKV(state, page, heads, width); err != nil {
				return err
			}
			var err error
			if d.cs, err = state.Driver.StreamCreate(strataRoundStreamNonBlocking); err != nil {
				return err
			}
		}
		// the six-row pass's buffers, as the drafter's load carves the ones its front reads
		maxT, stepRows := uint64(strataPrefillMaxT), uint64(strataDrafterStepSets*strataPrefillMaxT)
		carve := func(b *strataRoundBump) {
			d := groups
			d.dTok, d.dStep = b.take(maxT*strataRoundWordBytes), b.take(stepRows*strataDrafterStepWords*strataRoundWordBytes)
			d.dPos = b.take(stepRows * strataDrafterHeads * strataRoundWordBytes)
			d.rin, d.residual = b.take(maxT*hcn*strataRoundWordBytes), b.take(maxT*hcn*strataRoundWordBytes)
			d.emb, d.en, d.e2 = b.take(maxT*n*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes), b.take(maxT*n*strataRoundWordBytes)
			d.hn, d.h2 = b.take(maxT*hcn*strataRoundWordBytes), b.take(maxT*hcn*strataRoundWordBytes)
			d.mixed, d.lo = b.take(maxT*n*strataRoundWordBytes), b.take(maxT*strataGRRank*strataRoundWordBytes)
			d.rs, d.xn = b.take(maxT*strataGRStreams*strataRoundWordBytes), b.take(maxT*hcn*strataRoundWordBytes)
			d.xq = b.take(strataNativeQ81Size(int32(strataDrafterHeads*width), strataDrafterQ81Columns))
			d.kCur, d.vCur = b.take(maxT*kv*strataRoundWordBytes), b.take(maxT*kv*strataRoundWordBytes)
		}
		var count strataRoundBump
		carve(&count)
		var err error
		if groups.arena, err = e.resources.allocate(state, count.used); err != nil {
			return err
		}
		carve(&strataRoundBump{base: groups.arena})
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, d := range []*StrataDrafter{groups, batched} {
			if err := e.CloseStrataDrafter(context.Background(), d); err != nil {
				t.Error(err)
			}
		}
		if err := e.worker.Do(context.Background(), func(state *device.State) error {
			for _, upload := range uploads {
				if *upload.target != 0 {
					if err := state.Driver.MemFree(*upload.target); err != nil {
						return err
					}
				}
			}
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	next := make([]int32, strataBatchedCells)
	for index := range next {
		next[index] = int32(random.IntN(strataBatchedVocabRows))
	}
	if err := e.prefillStrataDrafterGroups(t.Context(), groups, rows, next, 0, 0); err != nil {
		t.Fatal(err)
	}
	if !batched.draftBatched() {
		t.Fatal("the drafter's Q8_0 projections and BF16 hyper-connection do not take the batched pass")
	}
	if err := e.prefillStrataDrafterBatch(t.Context(), batched, rows, next, 0, 0); err != nil {
		t.Fatal(err)
	}
	// cells reads a drafter's K or V rows, every cell's every head dequantized from its codes and its groups' scales.
	cells := func(d *StrataDrafter, values bool) []float32 {
		codes, scales := d.pools.key, d.pools.keyScale
		if values {
			codes, scales = d.pools.value, d.pools.valueScale
		}
		groupsPerRow := width / uint64(strataKVQ8Group)
		poolRows := uint64(d.slots) * heads * page
		codeBytes := make([]byte, poolRows*width)
		scaleBytes := make([]byte, poolRows*groupsPerRow*strataKVScaleBytes)
		if err := e.worker.Do(t.Context(), func(state *device.State) error {
			return errors.Join(state.Driver.MemcpyDtoH(codeBytes, codes), state.Driver.MemcpyDtoH(scaleBytes, scales))
		}); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, 0, strataBatchedCells*kv)
		for cell := range uint64(strataBatchedCells) {
			for head := range heads {
				row := ((cell/page)*heads+head)*page + cell%page
				for index := range width {
					scale := quant.Float16ToFloat32(binary.LittleEndian.Uint16(scaleBytes[(row*groupsPerRow+index/uint64(strataKVQ8Group))*strataKVScaleBytes:]))
					out = append(out, float32(int8(codeBytes[row*width+index]))*scale)
				}
			}
		}
		return out
	}
	for _, values := range []bool{false, true} {
		want, got := cells(groups, values), cells(batched, values)
		kind := "K"
		if values {
			kind = "V"
		}
		for row := range strataBatchedCells * int(heads) {
			var difference, norm float64
			for index := row * int(width); index < (row+1)*int(width); index++ {
				difference += math.Pow(float64(got[index]-want[index]), 2)
				norm += math.Pow(float64(want[index]), 2)
			}
			if norm == 0 || math.Sqrt(difference/norm) > strataBatchedTolerance {
				t.Fatalf("cell %d head %d: the batched %s row parts from the six-row pass's by %.4f of it (norm %.3g), want at most %.2f",
					row/int(heads), row%int(heads), kind, math.Sqrt(difference/math.Max(norm, math.SmallestNonzeroFloat64)), math.Sqrt(norm),
					strataBatchedTolerance)
			}
		}
	}
}
