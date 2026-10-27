//go:build windows

package executor

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// TestStrataRoundElementwiseMatchesStrata holds the decode round's elementwise kernels (strata_round_elementwise.cuh)
// to their definitions, bit for bit: the Q8_0 embedding gather gives each token's row of the table dequantized; the
// stream broadcast copies each token's row into its four streams; the copies from mapped memory copy their floats and
// words, the rows copy writing +0.0 into the plan's hit rows; and the experts' q8_1 quantizer (iq_kernels.cu's
// q8_1_store, IEEE arithmetic) writes each 32-value block's fp16 scale (its largest magnitude over 127) and sum (the
// warp's xor butterfly's order) and its codes (round half away from zero, clamped to 127) as a Go transcription does.
func TestStrataRoundElementwiseMatchesStrata(t *testing.T) {
	const (
		vocabulary = 50
		embedding  = 2560
		streams    = 4
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(43, 47))
	table := strataRandomWeights(t, draws, dtype.Q8_0, embedding, vocabulary, 1)
	rows, err := quant.Dequantize(dtype.Q8_0, table, uint64(embedding*vocabulary))
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{3, 7, 49, 0}
	tableBuffer, tokenBuffer := buffers.upload(t, table), buffers.upload(t, int32Bytes(tokens))
	gathered := buffers.zeros(t, 4*embedding*len(tokens))
	broadcast := buffers.zeros(t, 4*embedding*streams*len(tokens))
	rowBytes := uint64(len(table) / vocabulary)
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataEmbedRowsQ80(state, functions, tableBuffer, rowBytes, tokenBuffer, int32(len(tokens)), embedding, gathered); err != nil {
			return err
		}
		return launchStrataBroadcastStreams(state, functions, gathered, broadcast, embedding, streams, int32(len(tokens)))
	})
	got := float32s(buffers.read(t, gathered, 4*embedding*len(tokens)))
	wide := float32s(buffers.read(t, broadcast, 4*embedding*streams*len(tokens)))
	for index, token := range tokens {
		want := rows[int(token)*embedding : (int(token)+1)*embedding]
		if !slices.Equal(got[index*embedding:(index+1)*embedding], want) {
			t.Fatalf("token %d's gathered row differs from the table's", token)
		}
		for stream := range streams {
			at := (index*streams + stream) * embedding
			if !slices.Equal(wide[at:at+embedding], want) {
				t.Fatalf("token %d's stream %d differs from its row", token, stream)
			}
		}
	}
	// The copies from mapped memory.
	floats := make([]float32, 1024)
	for index := range floats {
		floats[index] = draws.Float32()
	}
	source := buffers.upload(t, float32Bytes(floats))
	target, words := buffers.zeros(t, 4*len(floats)), buffers.zeros(t, 4*len(floats))
	hits := buffers.upload(t, int32Bytes([]int32{2, 5}))
	count := buffers.upload(t, int32Bytes([]int32{2}))
	rowsTarget := buffers.zeros(t, 4*len(floats))
	const width = 128
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataCopyFromMapped(state, functions, target, source, int64(len(floats))); err != nil {
			return err
		}
		if err := launchStrataCopyI32FromMapped(state, functions, words, source, int64(len(floats))); err != nil {
			return err
		}
		return launchStrataCopyRowsFromMapped(state, functions, rowsTarget, source, int64(len(floats)/width), width, hits, count, 0, 0)
	})
	if !slices.Equal(float32s(buffers.read(t, target, 4*len(floats))), floats) ||
		!slices.Equal(float32s(buffers.read(t, words, 4*len(floats))), floats) {
		t.Fatal("a copy from mapped memory differs from its source")
	}
	copied := float32s(buffers.read(t, rowsTarget, 4*len(floats)))
	for row := range len(floats) / width {
		want := floats[row*width : (row+1)*width]
		if row == 2 || row == 5 {
			want = make([]float32, width)
		}
		if !slices.Equal(copied[row*width:(row+1)*width], want) {
			t.Fatalf("row %d of the rows copy differs", row)
		}
	}
	// The experts' q8_1 quantizer against its transcription.
	values := make([]float32, embedding*3)
	for index := range values {
		values[index] = float32(draws.NormFloat64()) * 3
	}
	values[64] = 0
	for index := 96; index < 128; index++ {
		values[index] = 0 // an all-zero block
	}
	valueBuffer := buffers.upload(t, float32Bytes(values))
	blocks := buffers.zeros(t, len(values)/strataNativeQ8K*strataNativeQ81Bytes)
	buffers.run(t, func(state *device.State, functions functionSet) error {
		return launchStrataQuantizeQ81Rows(state, functions, valueBuffer, 3, embedding, blocks)
	})
	if got, want := buffers.read(t, blocks, len(values)/strataNativeQ8K*strataNativeQ81Bytes), strataQ81Transcription(values); !slices.Equal(got, want) {
		for index := range want {
			if got[index] != want[index] {
				t.Fatalf("the q8_1 blocks differ from the transcription at byte %d (block %d)", index, index/strataNativeQ81Bytes)
			}
		}
	}
}

// TestStrataHeadPicksMatchStrata holds the head's multi-block argmax and the #879 guard (strata_round_elementwise.cuh,
// from verify_kernels.cu) to their definitions over Flash-Next's 248,320-token rows: each row's pick is its largest
// logit's index, the lower index on a tie (a tie planted across two blocks), the scratch left reusable for the next
// launch; the guard flags exactly the slices that hold a NaN or an infinity.
func TestStrataHeadPicksMatchStrata(t *testing.T) {
	const (
		rows       = 4
		vocabulary = 248320
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(79, 83))
	logits := make([]float32, rows*vocabulary)
	for index := range logits {
		logits[index] = float32(draws.NormFloat64())
	}
	// Row 1: the same largest value at two indices in different blocks; row 2: a NaN and an infinity in two slices.
	logits[vocabulary+9000], logits[vocabulary+200000] = 50, 50
	logits[2*vocabulary+77], logits[2*vocabulary+vocabulary-1] = float32(math.NaN()), float32(math.Inf(1))
	logitBuffer := buffers.upload(t, float32Bytes(logits))
	scratch := buffers.zeros(t, int(strataArgmaxScratchSize(rows)))
	picks, flags := buffers.zeros(t, 4*rows), buffers.zeros(t, 4*rows*strataNonfiniteBlocks)
	for range 2 { // the second launch reads the scratch the first left
		buffers.run(t, func(state *device.State, functions functionSet) error {
			if err := launchStrataArgmaxRows(state, functions, logitBuffer, rows, vocabulary, scratch, picks); err != nil {
				return err
			}
			return launchStrataLogitsNonfinite(state, functions, logitBuffer, rows, vocabulary, flags)
		})
		got := buffers.read(t, picks, 4*rows)
		for row := range rows {
			if row == 2 {
				continue // a NaN's comparisons pick nothing in particular
			}
			want := 0
			for index, value := range logits[row*vocabulary : (row+1)*vocabulary] {
				if value > logits[row*vocabulary+want] {
					want = index
				}
			}
			if pick := int(binary.LittleEndian.Uint32(got[4*row:])); pick != want {
				t.Fatalf("row %d picked %d, its largest logit is at %d", row, pick, want)
			}
		}
		flagged := buffers.read(t, flags, 4*rows*strataNonfiniteBlocks)
		perBlock := (vocabulary + strataNonfiniteBlocks - 1) / strataNonfiniteBlocks
		for row := range rows {
			for slice := range strataNonfiniteBlocks {
				want := uint32(0)
				if row == 2 && (slice == 77/perBlock || slice == (vocabulary-1)/perBlock) {
					want = 1
				}
				if got := binary.LittleEndian.Uint32(flagged[4*(row*strataNonfiniteBlocks+slice):]); got != want {
					t.Fatalf("row %d slice %d: the guard flagged %d, want %d", row, slice, got, want)
				}
			}
		}
	}
}

// strataQ81Transcription is q8_1_store in Go: a block's largest magnitude and sum over its 32 lanes by the warp's xor
// butterfly (offsets 16 to 1, each lane adding its partner's value), d = amax / 127, codes roundf(x / d) clamped to
// +-127 (zero for an all-zero block), the fp16 scale and sum.
func strataQ81Transcription(values []float32) []byte {
	const (
		lanes = 32
		step  = float32(127)
	)
	out := make([]byte, len(values)/lanes*strataNativeQ81Bytes)
	for block := range len(values) / lanes {
		var amax, sum [lanes]float32
		for lane := range lanes {
			v := values[block*lanes+lane]
			amax[lane], sum[lane] = float32(math.Abs(float64(v))), v
		}
		for offset := lanes / 2; offset > 0; offset >>= 1 {
			var nextMax, nextSum [lanes]float32
			for lane := range lanes {
				nextMax[lane] = max(amax[lane], amax[lane^offset])
				nextSum[lane] = sum[lane] + sum[lane^offset]
			}
			amax, sum = nextMax, nextSum
		}
		d := amax[0] / step
		at := block * strataNativeQ81Bytes
		binary.LittleEndian.PutUint16(out[at:], quant.Float32ToFloat16(d))
		binary.LittleEndian.PutUint16(out[at+2:], quant.Float32ToFloat16(sum[0]))
		for lane := range lanes {
			var code int8
			if amax[0] != 0 {
				q := float32(math.Round(float64(values[block*lanes+lane] / d)))
				code = int8(min(max(q, -step), step))
			}
			out[at+4+lane] = byte(code)
		}
	}
	return out
}
