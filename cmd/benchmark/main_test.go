package main

import (
	"testing"

	"overgo/internal/cuda/driver"
)

func TestParseOptionsBounds(t *testing.T) {
	want := struct {
		tokens, runs  int
		model, prompt string
	}{4, 3, "model.gguf", "hello"}
	options, err := parseOptions([]string{"-tokens", "4", "-runs", "3", "-warmup", "0", "model.gguf", "hello"})
	if err != nil {
		t.Fatal(err)
	}
	got := struct {
		tokens, runs  int
		model, prompt string
	}{options.Tokens, options.Runs, options.Model, options.Prompt}
	if got != want {
		t.Fatalf("options = %+v, want %+v", got, want)
	}
	for _, args := range [][]string{
		{"model.gguf"},
		{"-tokens", "0", "model.gguf", "hello"},
		{"-runs", "0", "model.gguf", "hello"},
		{"-warmup", "-1", "model.gguf", "hello"},
		{"-preload", "-native-quant", "model.gguf", "hello"},
		{"-host-cache", "-preload", "model.gguf", "hello"},
		{"-host-cache", "-native-quant", "model.gguf", "hello"},
		{"-batch-sequences", "-1", "model.gguf", "hello"},
		{"-batch-sequences", "1", "-cache-prompt", "model.gguf", "hello"},
		{"-batch-sequences", "1", "-speculative", "model.gguf", "hello"},
		{"-temperature", "NaN", "model.gguf", "hello"},
		{"-temperature", "+Inf", "model.gguf", "hello"},
		{"-temperature", "1e100", "model.gguf", "hello"},
	} {
		if _, err := parseOptions(args); err == nil {
			t.Fatalf("arguments %v were accepted", args)
		}
	}
}

func TestSubtractExecutionStats(t *testing.T) {
	want := driver.ExecutionStats{
		KernelLaunches: 7, StreamSynchronizations: 2,
		HostToDeviceBytes: 256, DeviceToDeviceBytes: 32,
	}
	before := driver.ExecutionStats{
		KernelLaunches:         10,
		StreamSynchronizations: 4,
		HostToDeviceBytes:      100,
		DeviceToDeviceBytes:    12,
	}
	after := before
	after.KernelLaunches += want.KernelLaunches
	after.StreamSynchronizations += want.StreamSynchronizations
	after.HostToDeviceBytes += want.HostToDeviceBytes
	after.DeviceToDeviceBytes += want.DeviceToDeviceBytes
	got := subtractExecutionStats(after, before)
	if got != want {
		t.Fatalf("delta = %+v, want %+v", got, want)
	}
}

func TestSummarizeRuns(t *testing.T) {
	want := summaryMetrics{
		TTFTMillisecondsP50: 20, TTFTMillisecondsMinimum: 10,
		TotalMillisecondsP50: 80, DecodeTokensPerSecondP50: 2,
		EndToEndTokensPerSecondP50: 5,
	}
	summary := summarizeRuns([]runMetrics{
		{TTFTMilliseconds: 30, TotalMilliseconds: 90, DecodeTokensPerSecond: 3, EndToEndTokensPerSecond: 6},
		{TTFTMilliseconds: 10, TotalMilliseconds: 70, DecodeTokensPerSecond: 1, EndToEndTokensPerSecond: 4},
		{TTFTMilliseconds: 20, TotalMilliseconds: 80, DecodeTokensPerSecond: 2, EndToEndTokensPerSecond: 5},
	})
	if summary != want {
		t.Fatalf("summary = %+v, want %+v", summary, want)
	}
}
