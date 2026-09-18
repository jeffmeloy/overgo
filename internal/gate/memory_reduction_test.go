package gate

import (
	"fmt"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/testevidence"
)

// TestGateDuplicateLoadRemoved holds the device lane's contract verification to
// one parse of the step output: every declared contract is checked against a
// single parsed report through testevidence.VerifyGoTestNames rather than
// re-parsing the whole output once per contract name.
func TestGateDuplicateLoadRemoved(t *testing.T) {
	var stream strings.Builder
	event := func(format string, args ...any) { fmt.Fprintf(&stream, format+"\n", args...) }
	const pkg = "overgo/internal/kern"
	for _, name := range []string{"TestAlphaKernel", "TestBetaKernel"} {
		event(`{"Action":"run","Package":%q,"Test":%q}`, pkg, name)
		event(`{"Action":"output","Package":%q,"Test":%q,"Output":"--- PASS\n"}`, pkg, name)
		event(`{"Action":"pass","Package":%q,"Test":%q,"Elapsed":0.1}`, pkg, name)
	}
	event(`{"Action":"pass","Package":%q,"Elapsed":0.3}`, pkg)
	output := stream.String()

	if err := testevidence.VerifyGoTestNames(output, false, []string{"TestAlphaKernel", "TestBetaKernel"}); err != nil {
		t.Fatalf("declared contracts rejected from one parse: %v", err)
	}
	if err := testevidence.VerifyGoTestNames(output, false, []string{"TestMissingKernel"}); err == nil {
		t.Fatal("a missing contract was accepted")
	}
}

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
