package gate

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/clioptions"
)

func TestGateTestFailureDiagnostics(t *testing.T) {
	repo := t.TempDir()
	for name, content := range map[string]string{
		"go.mod": "module failurefixture\n\ngo 1.23\n",
		"failure_test.go": `package failurefixture
import ("strings"; "testing")
func TestBroken(t *testing.T) { t.Error("expected retained owning assertion, got wrong value") }
func TestLater(t *testing.T) {
 for range 100 { t.Log(strings.Repeat("irrelevant later output ", 100)) }
}
`,
	} {
		if err := os.WriteFile(filepath.Join(repo, name), []byte(content), clioptions.OutputFileMode); err != nil {
			t.Fatal(err)
		}
	}
	for _, short := range []bool{true, false} {
		t.Run(fmt.Sprintf("short=%t", short), func(t *testing.T) {
			report, err := runGoTests(t.Context(), repo, []string{"./..."}, short, nil)
			if err == nil {
				t.Fatal("failing subprocess accepted")
			}
			for _, want := range []string{"failurefixture: TestBroken", "expected retained owning assertion, got wrong value"} {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("missing %q in %v", want, err)
				}
			}
			if strings.Contains(err.Error(), "irrelevant later output") || len(err.Error()) > 2*clioptions.DiagnosticTailBytes {
				t.Fatalf("unbounded or unrelated output in error: %v", err)
			}
			if report.PassedTests != 1 || len(report.Failed) != 2 {
				t.Fatalf("verdict evidence lost: %+v", report)
			}
		})
	}
}
