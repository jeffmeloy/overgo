//go:build windows

package executor

import (
	"encoding/binary"
	"math"
	"math/big"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// TestStrataNativeExpertGroupedMatchesStrata holds Strata's grouped native experts (strata_iq_experts.cuh, from
// iq_kernels.cu's native_expert_grouped) to a float64 transcription over Flash-Next's expert formats and shapes: each
// entry's gate and up rows are its blob's IQ3_S or IQ4_XS rows against its token's q8_1 activation; its SwiGLU row is
// stored as q8_1 blocks of their largest magnitude over 127 with codes within half a step; its down row is the blob's
// IQ4_NL or Q8_0 rows against those blocks, written to the entry's destination row. Groups past the device's count are
// not read, and a call striding all groups over one block row is bitwise the call with a row a group
// (native_grouped_parity).
func TestStrataNativeExpertGroupedMatchesStrata(t *testing.T) {
	const (
		embedding = strataExpertSub16Embedding
		ff        = strataExpertSubDownFF
		tokens    = 5
		groups    = 6
		capGroups = 8
		// bound is the relative L1 distance the products keep from the transcription.
		bound = 1e-5
		// weightScale sizes the weights as Flash-Next's experts.
		weightScale = 0.02
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(101, 103))
	// The largest magnitude a scale multiplies: IQ3_S's (1 + 2*15) * 15, IQ4_XS's 32 * 127, IQ4_NL's and Q8_0's 127.
	largest := map[dtype.Type]float32{dtype.IQ3S: 31 * 15, dtype.IQ4XS: 32 * 127, dtype.IQ4NL: 127, dtype.Q8_0: 127}
	randomRows := func(storage dtype.Type, rows, width int64) []byte {
		traits, _ := storage.Traits()
		blocks := uint64(rows*width) / traits.BlockSize
		data := make([]byte, blocks*traits.TypeSize)
		for index := range data {
			data[index] = byte(draws.Uint32())
		}
		for block := range blocks {
			scale := quant.Float32ToFloat16(weightScale * (draws.Float32() + 0.1) / largest[storage])
			binary.LittleEndian.PutUint16(data[block*traits.TypeSize:], scale)
		}
		return data
	}
	for _, formats := range []struct{ gu, down dtype.Type }{
		{dtype.IQ3S, dtype.IQ4NL}, {dtype.IQ4XS, dtype.IQ4NL}, {dtype.IQ3S, dtype.Q8_0},
	} {
		layout, err := strataNativeExpertLayoutOf(formats.gu, formats.down, embedding, ff)
		if err != nil {
			t.Fatal(err)
		}
		// Each group's entries: one a token, at most, its tokens in order; destinations a permutation.
		var grpStart, entTok []int32
		for range groups {
			grpStart = append(grpStart, int32(len(entTok)))
			for token := range int32(tokens) {
				if draws.IntN(2) == 0 || token == tokens-1 && int32(len(entTok)) == grpStart[len(grpStart)-1] {
					entTok = append(entTok, token)
				}
			}
		}
		entries := len(entTok)
		grpStart = append(grpStart, int32(entries))
		for len(grpStart) < capGroups+1 {
			grpStart = append(grpStart, int32(entries))
		}
		entDst := make([]int32, entries)
		for index, destination := range draws.Perm(entries) {
			entDst[index] = int32(destination)
		}
		capEntries := int64(entries + tokens)
		// The blobs, back to back on 256-byte boundaries; the groups past the count point at nothing.
		stride := strataExpertAlign(layout.Bytes)
		blobs := make([]byte, 0, groups*stride)
		gu, down := make([][]float32, groups), make([][]float32, groups)
		for group := range groups {
			blob := append(randomRows(formats.gu, 2*ff, embedding), randomRows(formats.down, embedding, ff)...)
			if uint64(len(blob)) != layout.Bytes {
				t.Fatalf("blob of %d bytes against the layout's %d", len(blob), layout.Bytes)
			}
			if gu[group], err = quant.Dequantize(formats.gu, blob[:layout.DownOff], 2*ff*embedding); err != nil {
				t.Fatal(err)
			}
			if down[group], err = quant.Dequantize(formats.down, blob[layout.DownOff:], embedding*ff); err != nil {
				t.Fatal(err)
			}
			blobs = append(blobs, blob...)
			blobs = append(blobs, make([]byte, stride-layout.Bytes)...)
		}
		blobBase := buffers.upload(t, blobs)
		var grpPtr []byte
		for group := range capGroups {
			address := uint64(0)
			if group < groups {
				address = uint64(blobBase) + uint64(group)*stride
			}
			grpPtr = binary.LittleEndian.AppendUint64(grpPtr, address)
		}
		x := make([]float32, tokens*embedding)
		for index := range x {
			x[index] = float32(draws.NormFloat64())
		}
		xBuffer := buffers.upload(t, float32Bytes(x))
		xq := buffers.zeros(t, tokens*embedding/strataNativeQ8K*strataNativeQ81Bytes)
		grpPtrBuffer, grpStartBuffer := buffers.upload(t, grpPtr), buffers.upload(t, int32Bytes(grpStart))
		count := buffers.upload(t, int32Bytes([]int32{groups}))
		entTokBuffer, entDstBuffer := buffers.upload(t, int32Bytes(entTok)), buffers.upload(t, int32Bytes(entDst))
		scratchBytes := int(strataNativeExpertScratchBytes(capEntries, ff))
		scratch, strideScratch := buffers.zeros(t, scratchBytes), buffers.zeros(t, scratchBytes)
		out, strideOut := buffers.zeros(t, 4*entries*embedding), buffers.zeros(t, 4*entries*embedding)
		buffers.run(t, func(state *device.State, functions functionSet) error {
			if err := launchStrataQuantizeQ81Rows(state, functions, xBuffer, tokens, embedding, xq); err != nil {
				return err
			}
			if err := launchStrataNativeExpertGrouped(state, functions, layout, grpPtrBuffer, grpStartBuffer, count,
				entDstBuffer, entTokBuffer, capGroups, capEntries, xq, scratch, out, 0); err != nil {
				return err
			}
			return launchStrataNativeExpertGrouped(state, functions, layout, grpPtrBuffer, grpStartBuffer, count,
				entDstBuffer, entTokBuffer, capGroups, capEntries, xq, strideScratch, strideOut, 1)
		})
		xValues := strataQ81Values(t, buffers.read(t, xq, tokens*embedding/strataNativeQ8K*strataNativeQ81Bytes), x)
		part := strataExpertAlign(uint64(capEntries) * ff * f32ByteSize)
		gotGate := float32s(buffers.read(t, scratch, 4*entries*ff))
		gotUp := float32s(buffers.read(t, scratch+driver.DevicePtr(part), 4*entries*ff))
		wantGate, wantUp := make([]float64, entries*ff), make([]float64, entries*ff)
		h := make([]float32, entries*ff)
		for group := range groups {
			for e := grpStart[group]; e < grpStart[group+1]; e++ {
				activation := xValues[int(entTok[e])*embedding : int(entTok[e]+1)*embedding]
				for r := range ff {
					var g, u float64
					for k := range embedding {
						g += float64(gu[group][r*embedding+k]) * activation[k]
						u += float64(gu[group][(ff+r)*embedding+k]) * activation[k]
					}
					wantGate[int(e)*ff+r], wantUp[int(e)*ff+r] = g, u
					gate := float64(gotGate[int(e)*ff+r])
					h[int(e)*ff+r] = float32(gate / (1 + math.Exp(-gate)) * float64(gotUp[int(e)*ff+r]))
				}
			}
		}
		if d := strataRelativeL1(float32To64(gotGate), wantGate); d > bound {
			t.Fatalf("%v/%v: the gate rows are %.3g from the transcription", formats.gu, formats.down, d)
		}
		if d := strataRelativeL1(float32To64(gotUp), wantUp); d > bound {
			t.Fatalf("%v/%v: the up rows are %.3g from the transcription", formats.gu, formats.down, d)
		}
		hValues := strataQ81Values(t, buffers.read(t, scratch+driver.DevicePtr(3*part), entries*ff/strataNativeQ8K*strataNativeQ81Bytes), h)
		wantOut := make([]float64, entries*embedding)
		for group := range groups {
			for e := grpStart[group]; e < grpStart[group+1]; e++ {
				for r := range embedding {
					var s float64
					for j := range ff {
						s += float64(down[group][r*ff+j]) * hValues[int(e)*ff+j]
					}
					wantOut[int(entDst[e])*embedding+r] = s
				}
			}
		}
		gotOut := buffers.read(t, out, 4*entries*embedding)
		if d := strataRelativeL1(float32To64(float32s(gotOut)), wantOut); d > bound {
			t.Fatalf("%v/%v: the down rows are %.3g from the transcription", formats.gu, formats.down, d)
		}
		if !slices.Equal(gotOut, buffers.read(t, strideOut, 4*entries*embedding)) {
			t.Fatalf("%v/%v: the call striding the groups over one block row differs", formats.gu, formats.down)
		}
	}
}

// TestStrataExpertTierMatchesStrata holds the rest of Strata's expert tier (strata_round_elementwise.cuh and
// strata_native_moe.cuh, from verify_kernels.cu, s2_expert_grouped.cu and native_moe.cu) to what each states, exactly: a
// flag already at its value releases the wait; fetch_blobs packs the first *n blobs in the order their pointers give
// and rebase_ptrs points the first *n pointers at them, the rest untouched; moe_hit_add adds hit rows to the rows its
// list names, others untouched; native_moe_combine_multi is each token's first product rounded, then one correctly
// rounded FMA an expert in order, then the shared row added.
func TestStrataExpertTierMatchesStrata(t *testing.T) {
	const (
		embedding = strataExpertSub16Embedding
		blobBytes = 65 * strataFetchBlobsWord
		blobs     = 5
		capacity  = 8
		rows      = 12
		tokens    = 3
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(107, 109))
	gauss := func(count int) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(draws.NormFloat64())
		}
		return values
	}
	source := make([]byte, capacity*blobBytes)
	for index := range source {
		source[index] = byte(draws.Uint32())
	}
	sourceBuffer := buffers.upload(t, source)
	order := draws.Perm(capacity)[:blobs]
	var pointers []byte
	for _, slot := range order {
		pointers = binary.LittleEndian.AppendUint64(pointers, uint64(sourceBuffer)+uint64(slot*blobBytes))
	}
	for len(pointers) < 8*capacity {
		pointers = binary.LittleEndian.AppendUint64(pointers, draws.Uint64())
	}
	pointerBuffer, count := buffers.upload(t, pointers), buffers.upload(t, int32Bytes([]int32{blobs}))
	packed := buffers.zeros(t, capacity*blobBytes)
	flag := buffers.upload(t, int32Bytes([]int32{7}))
	parts, hits := gauss(rows*embedding), gauss(rows*embedding)
	hitRows := []int32{3, 7, 0}
	partsBuffer, hitBuffer := buffers.upload(t, float32Bytes(parts)), buffers.upload(t, float32Bytes(hits))
	hitRowBuffer, hitCount := buffers.upload(t, int32Bytes(hitRows)), buffers.upload(t, int32Bytes([]int32{int32(len(hitRows))}))
	expertParts, weights, shared := gauss(tokens*strataCombineExperts*embedding), gauss(tokens*strataCombineExperts), gauss(tokens*embedding)
	expertBuffer, weightBuffer := buffers.upload(t, float32Bytes(expertParts)), buffers.upload(t, float32Bytes(weights))
	sharedBuffer, output := buffers.upload(t, float32Bytes(shared)), buffers.zeros(t, 4*tokens*embedding)
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataWaitFlagGE(state, functions, flag, 5); err != nil {
			return err
		}
		if err := launchStrataFetchBlobs(state, functions, pointerBuffer, count, packed, blobBytes, capacity); err != nil {
			return err
		}
		if err := launchStrataRebasePtrs(state, functions, pointerBuffer, count, packed, blobBytes); err != nil {
			return err
		}
		if err := launchStrataMoEHitAdd(state, functions, partsBuffer, hitBuffer, hitRowBuffer, hitCount, capacity, embedding); err != nil {
			return err
		}
		return launchStrataNativeMoECombineMulti(state, functions, expertBuffer, weightBuffer, sharedBuffer, output,
			embedding, strataCombineExperts, tokens)
	})
	gotPacked := buffers.read(t, packed, capacity*blobBytes)
	for index, slot := range order {
		if !slices.Equal(gotPacked[index*blobBytes:(index+1)*blobBytes], source[slot*blobBytes:(slot+1)*blobBytes]) {
			t.Fatalf("fetch_blobs: blob %d is not the one its pointer names", index)
		}
	}
	if !slices.Equal(gotPacked[blobs*blobBytes:], make([]byte, (capacity-blobs)*blobBytes)) {
		t.Fatal("fetch_blobs wrote past the count")
	}
	gotPointers := buffers.read(t, pointerBuffer, 8*capacity)
	for index := range capacity {
		want := binary.LittleEndian.Uint64(pointers[8*index:])
		if index < blobs {
			want = uint64(packed) + uint64(index*blobBytes)
		}
		if got := binary.LittleEndian.Uint64(gotPointers[8*index:]); got != want {
			t.Fatalf("rebase_ptrs: pointer %d is %#x, want %#x", index, got, want)
		}
	}
	wantParts := slices.Clone(parts)
	for _, row := range hitRows {
		for i := range embedding {
			wantParts[int(row)*embedding+i] += hits[int(row)*embedding+i]
		}
	}
	if !slices.Equal(float32s(buffers.read(t, partsBuffer, 4*rows*embedding)), wantParts) {
		t.Fatal("moe_hit_add: the rows differ from parts plus hits on the listed rows")
	}
	fma := func(a, b, c float32) float32 {
		exact := new(big.Float).SetPrec(200).SetFloat64(float64(a))
		exact.Mul(exact, new(big.Float).SetFloat64(float64(b))).Add(exact, new(big.Float).SetFloat64(float64(c)))
		rounded, _ := exact.Float32()
		return rounded
	}
	want := make([]float32, tokens*embedding)
	for token := range tokens {
		for col := range embedding {
			part := func(expert int) float32 {
				return expertParts[(token*strataCombineExperts+expert)*embedding+col]
			}
			sum := part(0) * weights[token*strataCombineExperts]
			for expert := 1; expert < strataCombineExperts; expert++ {
				sum = fma(part(expert), weights[token*strataCombineExperts+expert], sum)
			}
			want[token*embedding+col] = sum + shared[token*embedding+col]
		}
	}
	if !slices.Equal(float32s(buffers.read(t, output, 4*tokens*embedding)), want) {
		t.Fatal("native_moe_combine_multi differs from the ordered FMA chain")
	}
}
