package gate

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testutil"
)

// TestCampaignStructurePreflight rejects a failed or missing campaign check
// before opening the evidence store, even when the test command exits zero.
func TestCampaignStructurePreflight(t *testing.T) {
	t.Parallel()
	for _, outcome := range []string{"fail", "missing", "pass"} {
		t.Run(outcome, func(t *testing.T) {
			root := t.TempDir()
			testutil.WriteTextFile(t, root, "README.md", "")
			if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
				t.Fatal(err)
			}
			var output strings.Builder
			encoder := json.NewEncoder(&output)
			event := func(action, name string) {
				t.Helper()
				if err := encoder.Encode(map[string]string{"Action": action, "Package": "overgo/internal/plan", "Test": name}); err != nil {
					t.Fatal(err)
				}
			}
			event("start", "")
			for _, name := range []string{"TestSingleCanonicalCampaignPlan", "TestRSICampaignRatchetAndParallelStructure", "TestOptimizedValidationCampaign"} {
				action := "pass"
				if name == "TestOptimizedValidationCampaign" {
					if outcome == "missing" {
						continue
					}
					action = outcome
				}
				event("run", name)
				event(action, name)
			}
			event("pass", "")
			calls := 0
			g := gateContext{repo: root, storePath: "absent-store", paths: []string{plan.Path}}
			g.runCommand = func(_ string, name string, args ...string) (string, error) {
				calls++
				if name != "go" || !strings.Contains(strings.Join(args, " "), "test ./internal/plan -json -run") {
					t.Fatalf("unexpected preflight command %s %v", name, args)
				}
				return output.String(), nil
			}
			_, err := g.stepDocumentation()
			if calls != 1 || err == nil {
				t.Fatalf("commands=%d error=%v", calls, err)
			}
			if strings.Contains(err.Error(), "campaign structure:") != (outcome != "pass") {
				t.Fatalf("outcome=%s error=%v", outcome, err)
			}
			if _, err := os.Stat(filepath.Join(root, g.storePath)); !os.IsNotExist(err) {
				t.Fatalf("preflight touched the absent store: %v", err)
			}
		})
	}
}
