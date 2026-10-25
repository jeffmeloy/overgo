package server

import (
	"context"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/cuda/driver"
	"overgo/internal/modeldevice"
)

// monitoredGenerator serves the fake model on a card whose memory, experts
// and live load the test sets.
type monitoredGenerator struct {
	*memoryOwnersGenerator
	card []driver.CardReading
}

func (generator *monitoredGenerator) CardReadings(context.Context) ([]driver.CardReading, error) {
	generator.mu.Lock()
	defer generator.mu.Unlock()
	return slices.Clone(generator.card), nil
}

func (generator *monitoredGenerator) setCard(card []driver.CardReading) {
	generator.mu.Lock()
	generator.card = card
	generator.mu.Unlock()
}

// testMonitoredGenerator is a 24 GiB card holding a quarter of a model's
// experts, three in four routed reads served there, and a busy card's load;
// a prompt of three tokens holds one in the cache and takes 2 ms to prefill
// the other two.
func testMonitoredGenerator() *monitoredGenerator {
	memory := testDeviceMemory()
	memory.Experts = &modeldevice.ExpertPlacement{
		CardExperts: 1536, HostExperts: 6144, CardBytes: 6 * gibibyte, HostBytes: 24 * gibibyte, Routes: 4000, CardRoutes: 3000, LoadCardExperts: 512,
	}
	return &monitoredGenerator{
		memoryOwnersGenerator: &memoryOwnersGenerator{
			fakeGenerator: &fakeGenerator{promptCached: 1, promptDuration: 2 * time.Millisecond, tokenDelay: time.Millisecond},
			memory:        memory,
		},
		card: []driver.CardReading{
			{Name: "utilization_percent", Value: 87}, {Name: "temperature_celsius", Value: 64},
			{Name: "power_watts", Value: 312.5}, {Name: "power_limit_watts", Value: 425},
			{Name: "pcie_generation", Value: 4}, {Name: "pcie_width", Value: 16},
			{Name: "pcie_received_bytes_per_second", Value: 2.5e9},
		},
	}
}

// TestMetricsReportDeviceAndExpertState holds /metrics to the live state the
// runtime view's monitor reads: the served card's total and free memory, its
// load as the card's management library reads it, where a file-backed
// model's experts are and how many routed reads found theirs on the card, and
// the prompt tokens prefilled and the time they took. A card the library does
// not read and a model with no file-backed experts leave their lines out;
// /metrics stays out of the request ring, which a monitor polling it once a
// second would otherwise fill; and the ring holds each generation's reused
// prompt tokens and its prefill and decode time.
func TestMetricsReportDeviceAndExpertState(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, testMonitoredGenerator())
	defer handler.Close()
	completion := serveTestRequest(handler, http.MethodPost, "/v1/completions", `{"model":"test-model","prompt":"hi","max_tokens":2,"temperature":0}`)
	if completion.Code != http.StatusOK {
		t.Fatalf("completion status = %d body=%s", completion.Code, completion.Body.String())
	}
	metrics := serveTestRequest(handler, http.MethodGet, "/metrics", "")
	lines := strings.Split(metrics.Body.String(), "\n")
	for _, line := range []string{
		"overgo_prompt_tokens_total 2",
		"overgo_prompt_seconds_total 0.002",
		"overgo_device_memory_total_bytes 25769803776",
		"overgo_device_memory_free_bytes 10737418240",
		"overgo_experts_card 1536",
		"overgo_experts_host 6144",
		"overgo_experts_card_bytes 6442450944",
		"overgo_experts_host_bytes 25769803776",
		"overgo_expert_routes_total 4000",
		"overgo_expert_routes_card_total 3000",
		"overgo_experts_load_card 512",
		"overgo_card_utilization_percent 87",
		"overgo_card_temperature_celsius 64",
		"overgo_card_power_watts 312.5",
		"overgo_card_power_limit_watts 425",
		"overgo_card_pcie_generation 4",
		"overgo_card_pcie_width 16",
		"overgo_card_pcie_received_bytes_per_second 2.5e+09",
	} {
		if !slices.Contains(lines, line) {
			t.Errorf("metrics lack %q", line)
		}
	}
	if t.Failed() {
		t.Fatalf("metrics status %d:\n%s", metrics.Code, metrics.Body.String())
	}

	// A card the management library does not read and a model with no file-backed experts report neither.
	plain := newTestHandler(t, &memoryOwnersGenerator{fakeGenerator: &fakeGenerator{}, memory: testDeviceMemory()})
	defer plain.Close()
	bare := serveTestRequest(plain, http.MethodGet, "/metrics", "").Body.String()
	if !strings.Contains(bare, "\novergo_device_memory_free_bytes 10737418240\n") || strings.Contains(bare, "overgo_expert") || strings.Contains(bare, "overgo_card_") {
		t.Errorf("a card with no load readings and a model with no file-backed experts reported:\n%s", bare)
	}

	// The ring holds the completion alone, with its reused tokens and its prefill and decode time.
	log := monitorLog(t, handler)
	if len(log.Requests) != 1 {
		t.Fatalf("the ring holds %+v, want the completion alone", log.Requests)
	}
	entry := log.Requests[0]
	if entry.PromptTokens != 3 || entry.CachedTokens != 1 || entry.CompletionTokens != 2 || entry.PrefillMS != 2 || entry.DecodeMS <= 0 || entry.DecodeMS > entry.DurationMS {
		t.Errorf("the completion's entry = %+v, want 3 prompt tokens of which 1 reused, 2 output, a 2 ms prefill and a decode time within the request's duration", entry)
	}
}
