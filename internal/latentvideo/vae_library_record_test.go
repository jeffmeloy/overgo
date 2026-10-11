package latentvideo

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestCodecPathRecordStatesTheRuleTheCodeApplies holds the record of the
// codec's two convolution paths to the code it describes: the size it says a
// clip takes the library at is the size the session routes a clip by, and the
// clips it says take the library, and the picture and the small clips it says
// do not, stand on those sides of it. It holds the record to say its figures
// are a quiet device's, since the record it replaced was taken beside an
// evaluation and its seconds were not. The figures themselves are a
// measurement, which TestLibraryCodecKeepsWithinTheKernelsMemoryAndLevels
// takes, and are not read back here.
func TestCodecPathRecordStatesTheRuleTheCodeApplies(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", "video-codec-convolution-paths.txt"))
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]string{}
	for line := range strings.Lines(string(raw)) {
		if key, value, found := strings.Cut(strings.TrimSpace(line), ": "); found && !strings.Contains(key, " ") {
			fields[key] = value
		}
	}
	stated, _, _ := strings.Cut(fields["threshold"], " ")
	if threshold, err := strconv.Atoi(stated); err != nil || threshold != libraryClipValues {
		t.Fatalf("the record says a clip takes the library at %q values (%v), and the session routes a clip at %d", stated, err, libraryClipValues)
	}
	if !strings.HasPrefix(fields["quiet"], "yes") {
		t.Fatalf("the record says of its device: %q; its figures are to be a quiet device's", fields["quiet"])
	}
	const rows, columns = 480, 832
	for frames, takes := range map[int]bool{1: false, 5: true, 9: true, 21: true, 81: true} {
		if values := frames * rows * columns; (values >= libraryClipValues) != takes {
			t.Fatalf("%d frames at %d by %d are %d values: the record says such a clip takes the library is %t, and the size the session routes at, %d, says otherwise", frames, columns, rows, values, takes, libraryClipValues)
		}
		if written := strconv.Itoa(frames * rows * columns); frames < 81 && !strings.Contains(strings.ReplaceAll(string(raw), ",", ""), written) {
			t.Fatalf("the record does not state the %s values of %d frames at %d by %d", written, frames, columns, rows)
		}
	}
	// The two clips of 5 frames the size was set between, and the small clips
	// accepted as the kernels draw them.
	for extents, takes := range map[[2]int]bool{{480, 640}: true, {448, 480}: false, {64, 64}: false, {16, 32}: false} {
		values := 5 * extents[0] * extents[1]
		if (values >= libraryClipValues) != takes {
			t.Fatalf("5 frames at %d by %d are %d values: the record says such a clip takes the library is %t, and the size the session routes at, %d, says otherwise", extents[1], extents[0], values, takes, libraryClipValues)
		}
		if written := strconv.Itoa(values); !strings.Contains(strings.ReplaceAll(string(raw), ",", ""), written) {
			t.Fatalf("the record does not state the %s values of 5 frames at %d by %d", written, extents[1], extents[0])
		}
	}
}
