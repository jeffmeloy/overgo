package main

import (
	"testing"

	"llamacpp2go/internal/cuda/driver"
)

func TestParseOptionsBounds(t *testing.T) {
	options, err := parseOptions([]string{"-tokens", "4", "-runs", "3", "-warmup", "0", "model.gguf", "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if options.Tokens != 4 || options.Runs != 3 || options.Model != "model.gguf" || options.Prompt != "hello" {
		t.Fatalf("options = %+v", options)
	}
	for _, args := range [][]string{
		{"model.gguf"},
		{"-tokens", "0", "model.gguf", "hello"},
		{"-runs", "0", "model.gguf", "hello"},
		{"-warmup", "-1", "model.gguf", "hello"},
		{"-preload", "-native-quant", "model.gguf", "hello"},
	} {
		if _, err := parseOptions(args); err == nil {
			t.Fatalf("arguments %v were accepted", args)
		}
	}
}

func TestSubtractExecutionStats(t *testing.T) {
	before := driver.ExecutionStats{
		KernelLaunches:         10,
		StreamSynchronizations: 4,
		HostToDeviceBytes:      100,
		DeviceToDeviceBytes:    12,
	}
	after := driver.ExecutionStats{
		KernelLaunches:         17,
		StreamSynchronizations: 6,
		HostToDeviceBytes:      356,
		DeviceToDeviceBytes:    44,
	}
	got := subtractExecutionStats(after, before)
	if got.KernelLaunches != 7 ||
		got.StreamSynchronizations != 2 ||
		got.HostToDeviceBytes != 256 ||
		got.DeviceToDeviceBytes != 32 {
		t.Fatalf("delta = %+v", got)
	}
}

func TestSummarizeRuns(t *testing.T) {
	summary := summarizeRuns([]runMetrics{
		{TTFTMilliseconds: 30, TotalMilliseconds: 90, DecodeTokensPerSecond: 3, EndToEndTokensPerSecond: 6},
		{TTFTMilliseconds: 10, TotalMilliseconds: 70, DecodeTokensPerSecond: 1, EndToEndTokensPerSecond: 4},
		{TTFTMilliseconds: 20, TotalMilliseconds: 80, DecodeTokensPerSecond: 2, EndToEndTokensPerSecond: 5},
	})
	if summary.TTFTMillisecondsP50 != 20 ||
		summary.TTFTMillisecondsMinimum != 10 ||
		summary.TotalMillisecondsP50 != 80 ||
		summary.DecodeTokensPerSecondP50 != 2 ||
		summary.EndToEndTokensPerSecondP50 != 5 {
		t.Fatalf("summary = %+v", summary)
	}
}
