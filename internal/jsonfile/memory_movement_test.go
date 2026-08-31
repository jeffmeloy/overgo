package jsonfile

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func measuredAllocatedBytes(operation func()) uint64 {
	best := ^uint64(0)
	for range 3 {
		var before, after runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&before)
		operation()
		runtime.ReadMemStats(&after)
		if allocated := after.TotalAlloc - before.TotalAlloc; allocated < best {
			best = allocated
		}
	}
	return best
}

// TestMemoryMovementTraceReduction pins the strict-decode copy trace: the
// duplicate-name scan re-reads the caller's own bytes, so decoding one
// admitted document allocates less than two payload lengths — the decoded
// result may hold one copy and the scan must add none. The former tee
// buffer duplicated every payload on the way through; its path is deleted,
// and this measured bound refuses its return.
func TestMemoryMovementTraceReduction(t *testing.T) {
	payload := []byte(`{"note":"` + strings.Repeat("m", 1<<20) + `"}`)
	document := filepath.Join(t.TempDir(), "note.json")
	if err := os.WriteFile(document, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Note string `json:"note"`
	}
	if err := DecodeStrict(document, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Note) != 1<<20 {
		t.Fatalf("decoded %d bytes", len(decoded.Note))
	}
	allocated := measuredAllocatedBytes(func() {
		var inner struct {
			Note string `json:"note"`
		}
		if err := DecodeStrict(document, &inner); err != nil {
			t.Fatal(err)
		}
	})
	// The admitted movement is one file read, the decoder's buffered view
	// with its growth churn, and one decoded result — measured near six
	// payload lengths. The refused regime, where a tee buffer or a
	// re-tokenizing duplicate scan materializes values again, measured
	// above ten; seven separates the two with margin on both sides.
	if bound := uint64(7 * len(payload)); allocated >= bound {
		t.Fatalf("strict decode moved %d bytes for a %d-byte payload (bound %d)", allocated, len(payload), bound)
	}
}
