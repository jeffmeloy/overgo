package gate

import (
	"testing"

	"overgo/internal/clioptions"
)

// TestGateRetainedMemoryReduced holds the gate lane runner's output retention
// bounded: superviseLane streams a lane's whole stdout and stderr through a
// clioptions.TailWriter instead of a bytes.Buffer, so peak retained memory is
// the diagnostic tail, not the entire multi-package stream. A megabyte of lane
// output must never leave more than the tail retained.
func TestGateRetainedMemoryReduced(t *testing.T) {
	writer := clioptions.NewTailWriter(clioptions.DiagnosticTailBytes)
	chunk := make([]byte, 4096)
	for i := range chunk {
		chunk[i] = byte('A' + i%26)
	}
	for written := 0; written < 1<<20; written += len(chunk) {
		if _, err := writer.Write(chunk); err != nil {
			t.Fatal(err)
		}
		if writer.Len() > clioptions.DiagnosticTailBytes {
			t.Fatalf("lane retention %d exceeded the %d-byte diagnostic bound after %d bytes",
				writer.Len(), clioptions.DiagnosticTailBytes, written)
		}
	}
	if len(writer.Tail()) > clioptions.DiagnosticTailBytes+len("...") {
		t.Fatalf("reported tail %d exceeds the bound", len(writer.Tail()))
	}
}
