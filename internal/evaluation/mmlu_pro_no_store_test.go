package evaluation

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestTheMMLUProIdentityChecksReadNoStore runs the host gate's MMLU-Pro identity checks with the data root pointed at
// an empty directory and holds them to pass with nothing skipped (identity-gate/do, SQA-368): they read the pinned
// fixtures alone. A check that read the ambient store skipped where it found none and, where an import had changed the
// store's MMLU-Pro cases, refused landings that touched nothing of MMLU-Pro.
func TestTheMMLUProIdentityChecksReadNoStore(t *testing.T) {
	t.Parallel()
	check := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestMMLUProIsAGroupedChoiceWithItsIdentitiesUnchanged$", "-test.v", "-test.count=1")
	check.Env = append(os.Environ(), "OVERGO_DATA_ROOT="+t.TempDir())
	output, err := check.CombinedOutput()
	said := string(output)
	if err != nil || !strings.Contains(said, "--- PASS: TestMMLUProIsAGroupedChoiceWithItsIdentitiesUnchanged") || strings.Contains(said, "--- SKIP") {
		t.Fatalf("the MMLU-Pro identity checks with an empty data root: %v\n%s\nwant them to pass with nothing skipped", err, said)
	}
}
