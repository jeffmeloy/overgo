//go:build windows

package processmeasure

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/testcheck"
)

// readUseNow reads the process's use, failing the test where it is not measured.
func readUseNow(t *testing.T) Use {
	t.Helper()
	use, measured, err := ReadUse()
	if err != nil || !measured {
		t.Fatalf("the process's use read %+v (measured %t): %v; want it measured", use, measured, err)
	}
	return use
}

// TestAPartsUseIsItsOwn holds a ledger's processor time and bytes read to the parts that spent them, as its walls are
// (served-parts/do). An outer part whose inner part reads a file of 8 MiB and spins the processor has the reads and the
// processor time in the inner part's use, not its own as well; each part says it was measured; and the two parts' use
// sums to no more than the process's over the outer part. The MOSS clock had read processor time and bytes read beside
// the ledger by hand, and the decode's parts, read on the ledger, wrote zero for both, as if measured (SQA-471).
func TestAPartsUseIsItsOwn(t *testing.T) {
	t.Parallel()
	const size = 8 << 20
	path := filepath.Join(t.TempDir(), "read.bin")
	testcheck.NoError(t, os.WriteFile(path, make([]byte, size), 0o600))
	ctx, stages := WithStages(t.Context())
	before := readUseNow(t)
	outer := Stage(ctx, "outer")
	inner := Stage(ctx, "inner")
	if data := testcheck.Must(os.ReadFile(path))(t); len(data) != size {
		t.Fatalf("read %d bytes; want %d", len(data), size)
	}
	// The spin spends processor time the process's counters can read, whose grain is a scheduler tick.
	for start := readUseNow(t); readUseNow(t).CPU-start.CPU < 50*time.Millisecond; {
	}
	inner()
	outer()
	spent := readUseNow(t).minus(before, Use{})
	parts := testcheck.Must(stages.Parts())(t)
	if len(parts) != 2 || parts[0].Use == nil || parts[1].Use == nil {
		t.Fatalf("the parts are %+v; want inner and outer, each with its use", parts)
	}
	in, out := *parts[0].Use, *parts[1].Use
	if in.ReadBytes < size || out.ReadBytes >= size || in.CPU < 50*time.Millisecond || out.CPU >= in.CPU {
		t.Errorf("the inner part spent %+v and the outer %+v of its own; want the file's %d bytes and the spin's processor time in the inner part alone", in, out, size)
	}
	if in.ReadBytes+out.ReadBytes > spent.ReadBytes || in.CPU+out.CPU > spent.CPU {
		t.Errorf("the parts spent %+v and %+v, past the process's %+v over them", in, out, spent)
	}
}
