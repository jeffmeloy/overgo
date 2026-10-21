//go:build windows

package modeldevice

import (
	"bytes"
	"math/rand/v2"
	"testing"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestExpertPromotionsCopyBesideTheRunsOnTheCard drives a residency of two
// banks of three experts, two of each on the card, through the driver: a miss
// with nothing staged swaps at once and stages the coldest card expert on the
// residency's own stream; once staged, a miss takes that expert's slot, which
// goes to its staged copy, while its own copy to the card is in flight. After
// every step each table entry on the device names its expert's bytes.
func TestExpertPromotionsCopyBesideTheRunsOnTheCard(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	f := writeFixture(t, rand.New(rand.NewPCG(19, 31)), []fixtureTensor{
		{name: "gate", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
		{name: "down", shape: []uint64{16, 16, 3}, kind: dtype.F32},
	}, 0, 0)
	file := f.parse(t)
	const (
		experts                = 3
		tableBytes             = 2 * experts * 8
		gateStride, downStride = 256, 1024
	)
	worker, err := device.New(modelTestDeviceOrdinal)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	weights, err := NewDeviceWeights(worker)
	if err != nil {
		t.Fatal(err)
	}
	loader := NewLoader(file)
	loader.Experts(weights, file.Tensors, tableBytes+2*(gateStride+downStride))
	if err := loader.Load(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	cache := weights.experts
	builder := tensor.NewBuilder()
	nodes := []*tensor.Tensor{
		builder.Input("gate", file.Tensors[0].Type, tensor.MustShape(file.Tensors[0].Shape[:file.Tensors[0].Dimensions]...)),
		builder.Input("down", file.Tensors[1].Type, tensor.MustShape(file.Tensors[1].Shape[:file.Tensors[1].Dimensions]...)),
	}
	// step routes indices, runs work on the worker, and holds the device's tables and the experts they name to the
	// file, and the card to hold card.
	step := func(t *testing.T, card []uint32, work func(state *device.State) error, indices ...uint32) {
		t.Helper()
		err := worker.Do(t.Context(), func(state *device.State) error {
			flags, err := weights.ExpertRoutes(state, nodes)
			if err != nil {
				return err
			}
			for _, index := range indices {
				cache.routed[uint64(flags-cache.routedMap)+uint64(index)] = 1
			}
			if err := work(state); err != nil {
				return err
			}
			for _, name := range []string{"gate", "down"} {
				bank := cache.banks[name]
				table := make([]driver.DevicePtr, experts)
				if err := state.Driver.MemcpyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&table[0])), experts*8), bank.table); err != nil {
					return err
				}
				for index := range uint32(experts) {
					want := f.payloads[name][uint64(index)*bank.bytes : uint64(index+1)*bank.bytes]
					actual := bank.hosted[index]
					if actual == nil {
						actual = make([]byte, bank.bytes)
						if err := state.Driver.MemcpyDtoH(actual, table[index]); err != nil {
							return err
						}
					}
					resident := cache.banks[name].entries[index] != nil
					if table[index] != bank.pointers[index] || !bytes.Equal(actual[:bank.bytes], want) || resident != (bank.hosted[index] == nil) {
						t.Fatalf("%s expert %d does not read its bytes through the device's table", name, index)
					}
				}
				if got := onCard(cache, name); len(got) != len(card) {
					t.Fatalf("%s card holds %v, want %v", name, got, card)
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	advance := func(*device.State) error { return cache.advance(t.Context(), false) }
	landed := func(*device.State) error {
		if err := cache.land(t.Context(), true); err != nil {
			return err
		}
		return cache.publish(t.Context())
	}
	step(t, []uint32{1, 2}, advance, 2)
	if len(cache.staging) != 2 {
		t.Fatalf("%d card experts are staging, want expert 1 of each bank", len(cache.staging))
	}
	step(t, []uint32{1, 2}, landed)
	step(t, []uint32{2}, advance, 0)
	if len(cache.promoting) != 2 {
		t.Fatalf("%d promotions copy beside the next run, want expert 0 of each bank", len(cache.promoting))
	}
	step(t, []uint32{0, 2}, landed)
}
