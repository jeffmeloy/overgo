package server

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"sync"
	"testing"

	"overgo/internal/inference"
)

// memoryOwnersGenerator serves the fake model on a device whose memory by
// owner the test sets.
type memoryOwnersGenerator struct {
	*fakeGenerator
	mu     sync.Mutex
	memory inference.DeviceMemory
}

func (generator *memoryOwnersGenerator) DeviceMemory(context.Context) (inference.DeviceMemory, error) {
	generator.mu.Lock()
	defer generator.mu.Unlock()
	return generator.memory, nil
}

func (generator *memoryOwnersGenerator) set(memory inference.DeviceMemory) {
	generator.mu.Lock()
	generator.memory = memory
	generator.mu.Unlock()
}

const gibibyte = 1 << 30

// testDeviceMemory is a 24 GiB device holding the model's weights, cache
// and activations, 2 GiB of another process, and 10 GiB free.
func testDeviceMemory() inference.DeviceMemory {
	return inference.DeviceMemory{TotalBytes: 24 * gibibyte, FreeBytes: 10 * gibibyte, Owners: []inference.DeviceMemoryOwner{
		{Name: "weights", Bytes: 8 * gibibyte}, {Name: "KV cache and buffers", Bytes: 3 * gibibyte}, {Name: "activations", Bytes: 1 * gibibyte},
	}}
}

// TestLiveMemoryTiers: the health probe states the device's total and free
// memory and the served model's bytes by owner when the generator reports
// them, and only current and peak bytes when it does not.
func TestLiveMemoryTiers(t *testing.T) {
	t.Parallel()
	type health struct {
		Device struct {
			CurrentBytes uint64                        `json:"current_bytes"`
			TotalBytes   uint64                        `json:"total_bytes"`
			FreeBytes    uint64                        `json:"free_bytes"`
			Owners       []inference.DeviceMemoryOwner `json:"owners"`
		} `json:"device"`
	}
	probe := func(generator Generator) health {
		t.Helper()
		response := serveTestRequest(newTestHandler(t, generator), http.MethodGet, "/health", "")
		var decoded health
		if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil || response.Code != http.StatusOK {
			t.Fatalf("health %d %s: %v", response.Code, response.Body.String(), err)
		}
		return decoded
	}
	want := testDeviceMemory()
	owned := probe(&memoryOwnersGenerator{fakeGenerator: &fakeGenerator{}, memory: want})
	if owned.Device.TotalBytes != want.TotalBytes || owned.Device.FreeBytes != want.FreeBytes || !slices.Equal(owned.Device.Owners, want.Owners) {
		t.Fatalf("device with owners = %+v, want %+v", owned.Device, want)
	}
	plain := probe(&fakeGenerator{})
	if plain.Device.CurrentBytes == 0 || plain.Device.TotalBytes != 0 || plain.Device.Owners != nil {
		t.Fatalf("device without owners = %+v", plain.Device)
	}
}
