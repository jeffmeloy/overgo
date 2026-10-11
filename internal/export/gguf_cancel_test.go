package export

import (
	"bytes"
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
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

// TestFileWrittenUnderACancelledContextStops holds the one place an export and
// a merge create their model file to the context: a write cancelled after its
// first bytes gets the cancellation at its next write and the file keeps only
// what came before, and a file that already exists is refused, not replaced.
func TestFileWrittenUnderACancelledContextStops(t *testing.T) {
	t.Parallel()
	file := filepath.Join(t.TempDir(), "model.gguf")
	running, stop := context.WithCancel(t.Context())
	err := writeFile(running, file, func(destination io.Writer) error {
		if _, err := destination.Write([]byte("tensor")); err != nil {
			return err
		}
		stop()
		_, err := destination.Write([]byte("more"))
		return err
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("a write cancelled part way = %v, want the cancellation", err)
	}
	if kept, err := os.ReadFile(file); err != nil || string(kept) != "tensor" {
		t.Fatalf("the cancelled write left %q, %v", kept, err)
	}
	if err := writeFile(t.Context(), file, func(io.Writer) error { return nil }); !errors.Is(err, fs.ErrExist) {
		t.Fatalf("a write onto an existing file = %v, want it refused as existing", err)
	}
}
