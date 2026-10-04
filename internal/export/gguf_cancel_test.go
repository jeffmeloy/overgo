package export

import (
	"bytes"
	"context"
	"errors"
	"os"
	"testing"

	"overgo/internal/testutil"
)

// TestCancelledWritersStopAndLeaveNothing holds an export and a merge to the
// context they are given: a cancelled one refuses before it stages anything,
// the writer a running one writes through stops at the first write after the
// cancel and passes nothing on, and neither leaves a directory behind.
func TestCancelledWritersStopAndLeaveNothing(t *testing.T) {
	t.Parallel()
	source := testutil.HermeticLlamaGGUF(t, 64)
	root := t.TempDir()
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := GGUF(cancelled, source, root, "out", "f16"); !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled export = %v, want the cancellation", err)
	}
	if _, err := MergeGGUF(cancelled, source, []string{source}, []float32{0.5}, root, "merged"); !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled merge = %v, want the cancellation", err)
	}
	if entries, err := os.ReadDir(root); err != nil || len(entries) != 0 {
		t.Fatalf("a cancelled write left %v, %v", entries, err)
	}

	var written bytes.Buffer
	running, stop := context.WithCancel(t.Context())
	writer := contextWriter{ctx: running, writer: &written}
	if count, err := writer.Write([]byte("tensor")); count != len("tensor") || err != nil {
		t.Fatalf("a write before the cancel = %d, %v", count, err)
	}
	stop()
	if count, err := writer.Write([]byte("more")); count != 0 || !errors.Is(err, context.Canceled) || written.String() != "tensor" {
		t.Fatalf("a write after the cancel = %d, %v with %q written", count, err, written.String())
	}
}
