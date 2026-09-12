package gate

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestGateActiveTestBudget(t *testing.T) {
	for _, tc := range []struct {
		name, environment, persisted, want string
	}{
		{name: "active default", want: "0s"},
		{name: "operator environment", environment: "-timeout=1m", want: "1m0s"},
		{name: "operator test flag", environment: "-test.timeout=1m", want: "1m0s"},
		{name: "operator persisted configuration", persisted: "GOFLAGS=-timeout=2m\n", want: "2m0s"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := runtimeReaderFixture(t)
			goenv := filepath.Join(t.TempDir(), "goenv")
			if err := os.WriteFile(goenv, []byte(tc.persisted), 0600); err != nil {
				t.Fatal(err)
			}
			t.Setenv("GOENV", goenv)
			t.Setenv("GOFLAGS", tc.environment)
			source := fmt.Sprintf(`package readerclient
import ("flag"; "testing")
func TestBudget(t *testing.T) {
 if got := flag.Lookup("test.timeout").Value.String(); got != %q { t.Fatal(got) }
}
`, tc.want)
			if err := os.WriteFile(filepath.Join(g.repo, "internal/readerclient/client_test.go"), []byte(source), 0600); err != nil {
				t.Fatal(err)
			}
			report, err := g.runGoTestsAdmitted(t.Context(), []string{"./internal/readerclient"}, false, nil, false)
			if err != nil || !report.PackagePassed("overgo/internal/readerclient") {
				t.Fatalf("timeout contract failed: %v, %+v", err, report)
			}
		})
	}
}
