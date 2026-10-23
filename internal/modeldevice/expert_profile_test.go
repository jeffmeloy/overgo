package modeldevice

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestExpertProfile: a profile keeps its ranking through Strata's file; a load given one places the experts it ranks
// first on the card, each layer's banks together, and every expert still reads its own bytes; and the decode runs a
// residency counts, never a prompt's, rank the profile it answers.
func TestExpertProfile(t *testing.T) {
	// Layer 0 routed expert 2 five times, expert 0 once; layer 1 expert 1 twice.
	counts := [][]uint64{{1, 0, 5}, {0, 2, 0}}
	ranked := []ExpertPair{{0, 2}, {1, 1}, {0, 0}, {1, 0}, {0, 1}, {1, 2}}
	t.Run("file", func(t *testing.T) {
		profile := rankExpertsByRoutes(2, 3, func(layer, expert uint32) uint64 { return counts[layer][expert] })
		if !slices.Equal(profile.Pairs, ranked) {
			t.Fatalf("ranking = %v, want the routed most first and then the rest across the layers, %v", profile.Pairs, ranked)
		}
		path := filepath.Join(t.TempDir(), "profile.strp")
		if err := profile.Write(path); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		// The header's 24 bytes, then the pairs' four bytes each, then each layer's experts' ranks, as Strata writes them.
		const header, pair = 24, 4
		ranks := make([]int32, 6)
		if _, err := binary.Decode(data[header+6*pair:], binary.LittleEndian, ranks); err != nil || len(data) != header+6*pair+6*4 || !slices.Equal(ranks, []int32{2, 4, 0, 3, 1, 5}) {
			t.Fatalf("file of %d bytes holds ranks %v: %v", len(data), ranks, err)
		}
		read, err := ReadExpertProfile(path)
		if err != nil || read.Layers != 2 || read.Experts != 3 || !slices.Equal(read.Pairs, ranked) {
			t.Fatalf("read %+v: %v", read, err)
		}
		binary.LittleEndian.PutUint16(data[header:], 2)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadExpertProfile(path); err == nil {
			t.Fatal("a pair outside the profile's layers was read")
		}
	})
	f := writeFixture(t, rand.New(rand.NewPCG(23, 37)), []fixtureTensor{
		{name: "blk.0.ffn_gate_exps.weight", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
		{name: "blk.0.ffn_down_exps.weight", shape: []uint64{16, 16, 3}, kind: dtype.F32},
		{name: "blk.1.ffn_down_exps.weight", shape: []uint64{16, 16, 3}, kind: dtype.F32},
	}, 0, 0)
	file := f.parse(t)
	const (
		tableBytes             = 3 * 3 * 8
		gateStride, downStride = 256, 1024
		// Layer 0's experts 2 and 0, its banks together, then layer 1's expert 1.
		limit = tableBytes + 2*(gateStride+downStride) + downStride
	)
	open := func(t *testing.T, profile ExpertProfile) (*expertResidency, *fakeExpertDevice, error) {
		t.Helper()
		mem := newFakeExpertDevice()
		weights := mem.weights()
		loader := NewLoader(file)
		loader.Experts(weights, file.Tensors, limit)
		loader.RankExperts(profile)
		if err := loader.Load(t.Context(), nil); err != nil {
			return nil, mem, err
		}
		t.Cleanup(func() {
			if err := weights.experts.close(); err != nil {
				t.Error(err)
			}
		})
		return weights.experts, mem, nil
	}
	profile := ExpertProfile{Layers: 2, Experts: 3, Pairs: ranked}
	t.Run("ranked-fill", func(t *testing.T) {
		cache, mem, err := open(t, profile)
		if err != nil {
			t.Fatal(err)
		}
		// Layer 0 holds experts 0 and 2 on the card around expert 1 in host memory, whose bytes the file holds between.
		card := map[string][]uint32{"blk.0.ffn_gate_exps.weight": {0, 2}, "blk.0.ffn_down_exps.weight": {0, 2}, "blk.1.ffn_down_exps.weight": {1}}
		for name, want := range card {
			bank := cache.banks[name]
			for index := range uint32(3) {
				actual, err := mem.at(bank.pointers[index], bank.bytes)
				start := uint64(index) * bank.bytes
				if err != nil || !bytes.Equal(actual, f.payloads[name][start:start+bank.bytes]) {
					t.Fatalf("%s expert %d does not read its bytes: %v", name, index, err)
				}
			}
			if got := onCard(cache, name); !slices.Equal(got, want) {
				t.Fatalf("%s holds %v on the card, want %v", name, got, want)
			}
		}
		if cache.bytes > limit {
			t.Fatal("card bytes exceed the limit")
		}
		other := profile
		other.Experts = 4
		if _, mem, err := open(t, other); err == nil || mem.liveCount() != 0 {
			t.Fatalf("a profile of another geometry loaded, or left %d allocations: %v", mem.liveCount(), err)
		}
	})
	t.Run("decode-runs-rank", func(t *testing.T) {
		cache, _, err := open(t, profile)
		if err != nil {
			t.Fatal(err)
		}
		weights := &DeviceWeights{experts: cache}
		route := func(prompt bool, layer []string, index uint32) {
			t.Helper()
			builder := tensor.NewBuilder()
			var nodes []*tensor.Tensor
			for _, name := range layer {
				bank := cache.banks[name].info
				nodes = append(nodes, builder.Input(name, bank.Type, tensor.MustShape(bank.Shape[:bank.Dimensions]...)))
			}
			bitmap, err := weights.ExpertRoutes(nil, nodes)
			if err != nil {
				t.Fatal(err)
			}
			cache.routed[uint64(bitmap-cache.routedMap)+uint64(index)] = 1
			if err := cache.advance(t.Context(), prompt); err != nil {
				t.Fatal(err)
			}
		}
		first := []string{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"}
		second := []string{"blk.1.ffn_down_exps.weight"}
		for layer, banks := range [][]string{first, second} {
			for expert, runs := range counts[layer] {
				for range runs {
					route(false, banks, uint32(expert))
				}
			}
		}
		// A prompt's routes name most experts, and rank none.
		route(true, first, 0)
		got, err := weights.ExpertProfile(nil)
		if err != nil || got.Layers != 2 || got.Experts != 3 || !slices.Equal(got.Pairs, ranked) {
			t.Fatalf("profile %+v, want %v: %v", got, ranked, err)
		}
	})
}
