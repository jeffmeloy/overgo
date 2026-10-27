package inference

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// The round's draft layer (mtp.cpp) for Flash-Next: the draft file's layer as Strata's rt pack holds it, made at open
// from the same MTP file Strata's tools/mtp_rt.py reads (Strata at 296b59d2, MIT) and placed on the card before the
// trunk's experts, as Strata loads its drafter before it sizes its expert tier. dense.bin: every tensor but the experts
// and the vocabulary, in the file's name order, each 256-byte aligned - the large projections Q8_0 (numpy's rounding of
// their bf16 values, as mtp_rt.py writes them), the hyper-connections, router and shared expert gate bf16 as stored,
// the norms f32 with the Gemma one added; experts.bin: the 512 routed experts' Q2_0 rows relaid into S2 blobs (gate and
// up rows interleaved, the codes in one plane and the fp16 scales in another, a lossless relayout).

// The pack's geometry: the draft layer's experts (H, FF, NE), Q2_0's block, and the dense file's alignment.
const (
	strataDraftExperts     = 512
	strataDraftQ20Values   = 64 // a Q2_0 block's values: one fp16 scale and 16 code bytes
	strataDraftQ20Bytes    = 18
	strataDraftQ20Scale    = 2
	strataDraftDenseAlign  = 256
	strataDraftQ80Values   = 32
	strataDraftQ80Bytes    = 34
	strataDraftQ80Levels   = 127
	strataDraftRound       = 0.5
	strataDraftBF16Shift   = 16
	strataDraftExpertChunk = 32 // the experts read and relaid a write
	// strataDraftGateUp is an expert's gate and up rows, interleaved in its blob (slot 2r gate r, 2r + 1 up r).
	strataDraftGateUp = tensor.PairedExtent
	// a bf16's, an fp16's and a float's bytes
	strataDraftBF16Bytes = 2
	strataDraftF16Bytes  = 2
	strataDraftF32Bytes  = 4
)

// strataDraftQ80Names are mtp_rt.py's Q8: the projections the engine's multi-column MMVQ runs.
var strataDraftQ80Names = []string{
	"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
	"self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
	"mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight",
}

// strataDraftTensor is a dense.txt line: a tensor's kind (q8_0, bf16 or f32), its rows and columns, and its offset and
// bytes in dense.bin.
type strataDraftTensor struct {
	kind          string
	rows, cols    uint64
	offset, bytes uint64
}

// strataDraftPack is the draft layer on the card as the rt pack holds it, and its vocabulary.
type strataDraftPack struct {
	dense, experts *executor.DeviceBuffer
	denseBase      driver.DevicePtr
	expertsBase    driver.DevicePtr
	tensors        map[string]strataDraftTensor
	vocabulary     []int32
}

// strataDraftShortName is mtp_rt.py's short name: the file's name without mtp. and layers.0.
func strataDraftShortName(name string) string {
	short := strings.TrimPrefix(name, "mtp.")
	return strings.TrimPrefix(short, "layers.0.")
}

// strataDraftQ80 is mtp_rt.py's q8_0 over rows of bf16 values: 32-value blocks, d = amax / 127 as fp16, each value
// times 1 / d rounded half away from zero in float32 (numpy's sign * floor(|v| + 0.5)).
func strataDraftQ80(values []byte) []byte {
	count := len(values) / strataDraftBF16Bytes
	out := make([]byte, count/strataDraftQ80Values*strataDraftQ80Bytes)
	var block [strataDraftQ80Values]float32
	for b := range count / strataDraftQ80Values {
		amax := float32(0)
		for i := range block {
			bits := binary.LittleEndian.Uint16(values[(b*strataDraftQ80Values+i)*strataDraftBF16Bytes:])
			block[i] = math.Float32frombits(uint32(bits) << strataDraftBF16Shift)
			amax = max(amax, float32(math.Abs(float64(block[i]))))
		}
		d := amax / strataDraftQ80Levels
		inverse := float32(0)
		if d > 0 {
			inverse = 1 / d
		}
		destination := out[b*strataDraftQ80Bytes:]
		binary.LittleEndian.PutUint16(destination, quant.Float32ToFloat16(d))
		for i, value := range block {
			scaled := float32(value * inverse)
			rounded := float32(math.Floor(float64(float32(float32(math.Abs(float64(scaled))) + strataDraftRound))))
			if scaled < 0 {
				rounded = -rounded
			}
			destination[strataDraftF16Bytes+i] = byte(int8(rounded))
		}
	}
	return out
}

// strataDraftDense is mtp_rt.py's dense.bin and dense.txt over the draft file: its tensors but the experts and the
// vocabulary, by name, each converted and 256-byte aligned.
func strataDraftDense(file *gguf.File) ([]byte, map[string]strataDraftTensor, error) {
	var names []string
	for _, info := range file.Tensors {
		if strings.Contains(info.Name, "experts.") || info.Name == model.HyperDraftVocabulary {
			continue
		}
		names = append(names, info.Name)
	}
	slices.Sort(names)
	var dense []byte
	tensors := make(map[string]strataDraftTensor, len(names))
	for _, name := range names {
		info, _ := file.Tensor(name)
		data := make([]byte, info.Size)
		if err := file.ReadTensorData(info, data); err != nil {
			return nil, nil, err
		}
		short := strataDraftShortName(name)
		entry := strataDraftTensor{rows: 1, cols: info.Shape[0]}
		for _, dimension := range info.Shape[1:info.Dimensions] {
			entry.rows *= dimension
		}
		var raw []byte
		switch {
		case info.Type == dtype.F32: // the raw GemmaRMSNorm w -> 1 + w
			raw = make([]byte, len(data))
			for i := 0; i < len(data); i += strataDraftF32Bytes {
				value := math.Float32frombits(binary.LittleEndian.Uint32(data[i:])) + 1
				binary.LittleEndian.PutUint32(raw[i:], math.Float32bits(value))
			}
			entry.kind, entry.rows, entry.cols = "f32", 1, entry.rows*entry.cols
		case info.Type != dtype.BF16:
			return nil, nil, fmt.Errorf("strata draft: %s is %s, where the pack reads bf16", name, info.Type)
		case slices.Contains(strataDraftQ80Names, short):
			raw, entry.kind = strataDraftQ80(data), "q8_0"
		default:
			raw, entry.kind = data, "bf16"
		}
		dense = append(dense, make([]byte, (strataDraftDenseAlign-len(dense)%strataDraftDenseAlign)%strataDraftDenseAlign)...)
		entry.offset, entry.bytes = uint64(len(dense)), uint64(len(raw))
		dense = append(dense, raw...)
		tensors[short] = entry
	}
	return dense, tensors, nil
}

// strataDraftBlobBytes is an S2 blob's bytes: the gate and up rows' codes and scales and the down rows'.
func strataDraftBlobBytes() uint64 {
	gateUpRows, downRows := uint64(strataDraftGateUp*strataRoundExpertFF), uint64(strataRoundEmbedding)
	gateUpBlocks, downBlocks := uint64(strataRoundEmbedding/strataDraftQ20Values), uint64(strataRoundExpertFF/strataDraftQ20Values)
	return (gateUpRows*gateUpBlocks + downRows*downBlocks) * strataDraftQ20Bytes
}

// strataDraftBlobs is mtp_rt.py's blob_of for count experts: gateUp holds each expert's 1,280 gate-then-up Q2_0 rows,
// down its 2,560 down rows; each blob interleaves the gate and up rows (slot 2r gate r, 2r + 1 up r), then lays the
// interleaved rows' codes, the down rows' codes, the interleaved rows' scales and the down rows' scales back to back.
func strataDraftBlobs(gateUp, down []byte, count int) []byte {
	ff, h := strataRoundExpertFF, strataRoundEmbedding
	gateUpRow, downRow := h/strataDraftQ20Values*strataDraftQ20Bytes, ff/strataDraftQ20Values*strataDraftQ20Bytes
	blob := strataDraftBlobBytes()
	out := make([]byte, uint64(count)*blob)
	for e := range count {
		destination := out[uint64(e)*blob:]
		rows := gateUp[e*strataDraftGateUp*ff*gateUpRow:]
		downs := down[e*h*downRow:]
		slot := func(i int) []byte { // interleaved slot i: gate row i / 2, or up row (i - 1) / 2
			if i%strataDraftGateUp == 0 {
				return rows[(i/strataDraftGateUp)*gateUpRow:][:gateUpRow]
			}
			return rows[(ff+i/strataDraftGateUp)*gateUpRow:][:gateUpRow]
		}
		at := 0
		for i := range strataDraftGateUp * ff {
			for _, block := range slices.Collect(slices.Chunk(slot(i), strataDraftQ20Bytes)) {
				at += copy(destination[at:], block[strataDraftQ20Scale:])
			}
		}
		for r := range h {
			for _, block := range slices.Collect(slices.Chunk(downs[r*downRow:][:downRow], strataDraftQ20Bytes)) {
				at += copy(destination[at:], block[strataDraftQ20Scale:])
			}
		}
		for i := range strataDraftGateUp * ff {
			for _, block := range slices.Collect(slices.Chunk(slot(i), strataDraftQ20Bytes)) {
				at += copy(destination[at:], block[:strataDraftQ20Scale])
			}
		}
		for r := range h {
			for _, block := range slices.Collect(slices.Chunk(downs[r*downRow:][:downRow], strataDraftQ20Bytes)) {
				at += copy(destination[at:], block[:strataDraftQ20Scale])
			}
		}
	}
	return out
}

// openStrataDraftPack makes the pack from the draft file and places it on the card: dense.bin in one buffer, the
// experts' blobs in another, written a chunk of experts at a time.
func openStrataDraftPack(ctx context.Context, cuda *executor.Executor, file *gguf.File) (*strataDraftPack, error) {
	vocabulary, err := readDraftVocabulary(file)
	if err != nil {
		return nil, err
	}
	if len(vocabulary) == 0 {
		return nil, errors.New("strata draft: the draft file holds no draft vocabulary")
	}
	dense, tensors, err := strataDraftDense(file)
	if err != nil {
		return nil, err
	}
	pack := &strataDraftPack{tensors: tensors, vocabulary: make([]int32, len(vocabulary))}
	for index, id := range vocabulary {
		pack.vocabulary[index] = int32(id)
	}
	fail := func(err error) (*strataDraftPack, error) {
		return nil, errors.Join(err, pack.close(context.WithoutCancel(ctx)))
	}
	if pack.dense, pack.denseBase, err = strataDraftBuffer(ctx, cuda, uint64(len(dense))); err != nil {
		return fail(err)
	}
	if err := cuda.WriteDeviceBytes(ctx, []executor.DeviceBytes{{Pointer: pack.denseBase, Data: dense}}); err != nil {
		return fail(err)
	}
	gateUp, gateUpFound := file.Tensor("mtp.layers.0.mlp.experts.gate_up_proj")
	down, downFound := file.Tensor("mtp.layers.0.mlp.experts.down_proj")
	if !gateUpFound || !downFound || gateUp.Type != dtype.Q2_0 || down.Type != dtype.Q2_0 {
		return fail(errors.New("strata draft: the routed experts are not the Q2_0 banks the pack relays"))
	}
	blob := strataDraftBlobBytes()
	if pack.experts, pack.expertsBase, err = strataDraftBuffer(ctx, cuda, strataDraftExperts*blob); err != nil {
		return fail(err)
	}
	gateUpExpert, downExpert := gateUp.Size/strataDraftExperts, down.Size/strataDraftExperts
	for first := 0; first < strataDraftExperts; first += strataDraftExpertChunk {
		count := min(strataDraftExpertChunk, strataDraftExperts-first)
		rows := make([]byte, uint64(count)*gateUpExpert)
		downs := make([]byte, uint64(count)*downExpert)
		if err := file.ReadTensorRange(gateUp, uint64(first)*gateUpExpert, rows); err != nil {
			return fail(err)
		}
		if err := file.ReadTensorRange(down, uint64(first)*downExpert, downs); err != nil {
			return fail(err)
		}
		if err := cuda.WriteDeviceBytes(ctx, []executor.DeviceBytes{{
			Pointer: pack.expertsBase + driver.DevicePtr(uint64(first)*blob), Data: strataDraftBlobs(rows, downs, count),
		}}); err != nil {
			return fail(err)
		}
	}
	return pack, nil
}

// strataDraftBuffer is card memory of bytes and its address.
func strataDraftBuffer(ctx context.Context, cuda *executor.Executor, bytes uint64) (*executor.DeviceBuffer, driver.DevicePtr, error) {
	buffer, err := cuda.AllocateDeviceBuffer(ctx, bytes)
	if err != nil {
		return nil, 0, err
	}
	value, err := buffer.Value(tensor.MustShape((bytes + strataDraftF32Bytes - 1) / strataDraftF32Bytes))
	if err != nil {
		return nil, 0, errors.Join(err, buffer.Release(context.Background()))
	}
	return buffer, value.Pointer, nil
}

// close releases the pack's card memory.
func (p *strataDraftPack) close(ctx context.Context) error {
	if p == nil {
		return nil
	}
	return executor.ReleaseDeviceBuffers(ctx, p.dense, p.experts)
}

// weight is a dense tensor's address, refused where it is missing or not of kind.
func (p *strataDraftPack) weight(name, kind string) (driver.DevicePtr, error) {
	entry, found := p.tensors[name]
	if !found || entry.kind != kind {
		return 0, fmt.Errorf("mtp: %s is missing (%s)", name, kind)
	}
	return p.denseBase + driver.DevicePtr(entry.offset), nil
}
