package gate

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

func TestDocumentationCostAcceptance(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	runGitFixture(t, root, "init")
	testutil.WriteTextFile(t, root, "go.mod", "module failurefixture\n\ngo 1.26\n")
	testutil.WriteTextFile(t, root, "broken/fixture_test.go", `package broken
import "testing"
func TestFirst(t *testing.T) { t.Error("seeded failure") }
func TestLaterAcquisition(t *testing.T) { t.Fatal("later acquisition started") }
`)
	testutil.WriteTextFile(t, root, "sibling/fixture_test.go", `package sibling
import "testing"
func TestIndependent(t *testing.T) {}
`)
	for _, short := range []bool{true, false} {
		report, err := (&gateContext{repo: root}).runGoTests(t.Context(), []string{"./..."}, short, nil)
		if err == nil || !strings.Contains(err.Error(), "seeded failure") {
			t.Fatalf("lost initial failure: report=%+v error=%v", report, err)
		}
		if strings.Contains(err.Error(), "later acquisition started") || slices.Contains(report.Failed, "failurefixture/broken: TestLaterAcquisition") {
			t.Fatalf("started another acquisition after failure: %+v", report)
		}
		if !report.PackagePassed("failurefixture/sibling") {
			t.Fatalf("lost independent sibling pass: %+v", report)
		}
		if report.PackagePassed("failurefixture/broken") {
			t.Fatalf("credited failed or unfinished work: %+v", report)
		}
	}
}
