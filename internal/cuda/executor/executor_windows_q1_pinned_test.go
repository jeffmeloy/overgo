//go:build windows

package executor

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// q1PinnedOutputs: SHA-256 of each case's output bytes as the per-element
// mul_mat_q1_0_f32 kernel at 41c23b87c produced them on the
// compute-capability 8.9 device. The kernel that takes a block at a time adds
// each lane's products in the same order, so the bits hold; a case whose bits
// move has changed what a Q1_0 model computes.
var q1PinnedOutputs = map[string]string{
	"ffn-gate":           "2a8e72a7e192d7c39e0822dfae467a0b6e82df01707fd5190d608c5148274d81",
	"ffn-down":           "a7a7cd5e351cf84a150d0efae203b22e85dbc389678417bafe1fa3578e6cbef4",
	"delta-qkv":          "ad9f1da534d640d9a3ab2a000bd34cd0fdfa3257d6c202c206276f8666c54fd4",
	"delta-out":          "6832b2b04a4bb8108f589eed67c82809a14032cf468b3c4aafb02945330983b5",
	"delta-alpha":        "706a3cb3e1b878eee09b2c32dde53a8f68cf2f4ad6f498d3714d63cd8d318d14",
	"attention-key":      "e18fec210ed28125f1288291796493be8feb7ba46a0295171acdaf7215d90d8f",
	"head":               "f12901912b9b8650ac5f95862a17e44fdce1bfd479b10fb89150ed86dafb0dde",
	"one-block-one-row":  "7d9cea668cfe03624fe8339a443181cee788d0ddb7bb0fa2d37f5a1170261551",
	"two-blocks":         "4c8a0b7ffa49b3c54eef73922bb6f4c00d97f40e3e30edefd4a17cfbf2ebe212",
	"three-blocks":       "5eb3deb05cff9bb210ee567f699102a9e19cbf1d9a3ba7f8491df023d8708ed4",
	"five-blocks":        "f90beebc2a8465fb7abfdfcc49539c0ae26818f57d37d46167a0f3317effec6a",
	"seven-blocks":       "b425c9a78e231c2a32ed2061b77f148d83f927341cc1e8ee7c1b7d9f9e75fe50",
	"columns-5":          "551c1f21f79ad6dc9272538f8c295e868b61013c5002afa92a8bdb9f06b1205c",
	"columns-31":         "f2c23c8b65cc0e53930efbad21f4f4c1acb1db2b322f4e076bdb8b399d2e67d8",
	"columns-past-floor": "543728462ccfeb4aa0fbf6f27ceb62fe9d041e24a4bc6a6991efa37c9a5b8b15",
	"grouped":            "e4f39586c61b4d4195b258810e6d796e4c461e677255a3343bb3616da9a01dcc",
}

// TestExecutorQ1MulMatOutputBitsArePinned runs each case's node through the
// Q1_0 mul_mat kernel and holds the output to the pinned bits, and to the
// products evaluated in float64 from the block bytes.
func TestExecutorQ1MulMatOutputBitsArePinned(t *testing.T) {
	cuda := newFixtureExecutor(t)
	for _, pinned := range q1PinnedCases {
		t.Run(pinned.name, func(t *testing.T) {
			got := pinned.run(t, cuda)
			digest := sha256.Sum256(driver.Bytes(got))
			if output := hex.EncodeToString(digest[:]); output != q1PinnedOutputs[pinned.name] {
				t.Errorf("output %s, pinned %q", output, q1PinnedOutputs[pinned.name])
			}
		})
	}
}

// q1PinnedCase is one Q1_0 weight [inner, rows] against columns activation
// columns, through the graph nodes that launch mul_mat_q1_0_f32: mul_mat
// under the exact policy, which keeps the kernel at every column count, and
// grouped_mul_mat. The width is whole 128-column blocks; the launchers refuse
// any other.
type q1PinnedCase struct {
	name                 string
	inner, rows, columns uint64
	// groups: weight matrices of a grouped_mul_mat; zero for mul_mat.
	groups uint64
}

var q1PinnedCases = []q1PinnedCase{
	// Bonsai-27B's decode shapes, one column: the feed-forward pair, a
	// gated-delta layer's projections, an attention key and the output head.
	{name: "ffn-gate", inner: 5120, rows: 17408, columns: 1},
	{name: "ffn-down", inner: 17408, rows: 5120, columns: 1},
	{name: "delta-qkv", inner: 5120, rows: 10240, columns: 1},
	{name: "delta-out", inner: 6144, rows: 5120, columns: 1},
	{name: "delta-alpha", inner: 5120, rows: 48, columns: 1},
	{name: "attention-key", inner: 5120, rows: 1024, columns: 1},
	{name: "head", inner: 5120, rows: 248320, columns: 1},
	// The shortest chain a lane can take, in a block of one warp.
	{name: "one-block-one-row", inner: 128, rows: 1, columns: 1},
	// The compiled loop takes four blocks a pass and the rest one at a
	// time: block counts of every remainder, over row counts that fill
	// neither a warp's eight-row block nor a multiple of it.
	{name: "two-blocks", inner: 256, rows: 3, columns: 1},
	{name: "three-blocks", inner: 384, rows: 13, columns: 1},
	{name: "five-blocks", inner: 640, rows: 37, columns: 1},
	{name: "seven-blocks", inner: 896, rows: 9, columns: 2},
	// A short prompt: one warp per row and column.
	{name: "columns-5", inner: 5120, rows: 1024, columns: 5},
	{name: "columns-31", inner: 640, rows: 77, columns: 31},
	{name: "columns-past-floor", inner: 256, rows: 19, columns: 40},
	// The grouped node launches the kernel once per token and group.
	{name: "grouped", inner: 256, rows: 24, columns: 3, groups: 4},
}

// q1PinnedDraws: a splitmix64 stream keyed by the case name and the tensor's
// ordinal alone.
type q1PinnedDraws uint64

func q1PinnedStream(name string, ordinal uint64) q1PinnedDraws {
	state := ordinal
	for _, character := range []byte(name) {
		state = state*1099511628211 + uint64(character)
	}
	return q1PinnedDraws(state)
}

func (d *q1PinnedDraws) next() uint64 {
	*d += 0x9e3779b97f4a7c15
	draw := uint64(*d)
	draw = (draw ^ draw>>30) * 0xbf58476d1ce4e5b9
	draw = (draw ^ draw>>27) * 0x94d049bb133111eb
	return draw ^ draw>>31
}

// unit: a draw in [-1, 1).
func (d *q1PinnedDraws) unit() float32 { return float32(d.next()>>40)/(1<<23) - 1 }

const (
	q1PinnedBlockWidth = 128
	q1PinnedBlockBytes = 18
	// q1PinnedScale bounds a block's drawn scale: [0.004, 0.036).
	q1PinnedScale = 0.036
)

// q1PinnedWeights: blocks Q1_0 blocks, each a drawn f16 scale and 128 drawn
// sign bits.
func q1PinnedWeights(name string, blocks uint64) []byte {
	draws := q1PinnedStream(name, 1)
	storage := make([]byte, blocks*q1PinnedBlockBytes)
	for block := range blocks {
		base := block * q1PinnedBlockBytes
		scale := quant.Float32ToFloat16(0.02 + 0.016*draws.unit())
		storage[base], storage[base+1] = byte(scale), byte(scale>>8)
		low, high := draws.next(), draws.next()
		for index := range uint64(8) {
			storage[base+2+index] = byte(low >> (8 * index))
			storage[base+10+index] = byte(high >> (8 * index))
		}
	}
	return storage
}

// run executes the case's node, holds a spread of its outputs to the same
// products evaluated in float64 from the block bytes, and returns the device
// output.
func (c q1PinnedCase) run(t *testing.T, cuda *Executor) []float32 {
	t.Helper()
	matrices := max(c.groups, 1)
	storage := q1PinnedWeights(c.name, c.inner*c.rows*matrices/q1PinnedBlockWidth)
	draws := q1PinnedStream(c.name, 2)
	activation := make([]float32, c.inner*matrices*c.columns)
	for index := range activation {
		activation[index] = draws.unit()
	}
	builder := tensor.NewBuilder()
	var weight, input, output *tensor.Tensor
	if c.groups == 0 {
		weight = builder.Input("weight", dtype.Q1_0, tensor.MustShape(c.inner, c.rows))
		input = builder.Input("input", dtype.F32, tensor.MustShape(c.inner, c.columns))
		output = builder.MulMat(weight, input)
	} else {
		weight = builder.Input("weight", dtype.Q1_0, tensor.MustShape(c.inner, c.rows, c.groups))
		input = builder.Input("input", dtype.F32, tensor.MustShape(c.inner, c.groups, c.columns))
		output = builder.GroupedMulMat(weight, input)
	}
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	result, err := cuda.executeWithDeviceFeeds(
		context.WithoutCancel(t.Context()), []*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{input: {Shape: input.Shape, Data: activation}},
		map[*tensor.Tensor]driver.DevicePtr{weight: copyFixtureDeviceBytes(t, cuda.worker, storage)},
	)
	if err != nil {
		t.Fatal(err)
	}
	got := result[output].Data
	outputs := c.rows * matrices * c.columns
	if uint64(len(got)) != outputs {
		t.Fatalf("output has %d values, want %d", len(got), outputs)
	}
	// Output layout: [rows, columns] for mul_mat, [rows, groups, columns]
	// for the grouped node; the activation is [inner, groups, columns]. A
	// block is its f16 scale and 128 sign bits, a set bit the scale and a
	// clear bit its negation. A row's sum is of order the scale times the
	// root of its width, and the device's F32 chains stay within a
	// relative 1e-4 of that. The widest cases are checked on every
	// stride-th output and the last.
	const checked = 4096
	stride := max(outputs/checked, 1)
	rowBytes := c.inner / q1PinnedBlockWidth * q1PinnedBlockBytes
	tolerance := 1e-4 * q1PinnedScale * math.Sqrt(float64(c.inner))
	for index := uint64(0); index < outputs; index += stride {
		if index+stride >= outputs {
			index = outputs - 1
		}
		row, vector := index%c.rows, index/c.rows
		matrix := vector % matrices
		values := activation[vector*c.inner:][:c.inner]
		blocks := storage[(matrix*c.rows+row)*rowBytes:][:rowBytes]
		var want float64
		for block := range c.inner / q1PinnedBlockWidth {
			bytes := blocks[block*q1PinnedBlockBytes:][:q1PinnedBlockBytes]
			var signed float64
			for column, value := range values[block*q1PinnedBlockWidth:][:q1PinnedBlockWidth] {
				if bytes[2+column/8]>>(column%8)&1 != 0 {
					signed += float64(value)
				} else {
					signed -= float64(value)
				}
			}
			want += signed * float64(quant.Float16ToFloat32(uint16(bytes[0])|uint16(bytes[1])<<8))
		}
		if difference := math.Abs(float64(got[index]) - want); difference > tolerance {
			t.Fatalf("value[%d] = %v, want %v (difference %g)", index, got[index], want, difference)
		}
	}
	return got
}
