package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestRuntimeValueFlowAcceptance holds the runtime value-flow boundary: a path
// or command value that reaches a file-owner or subprocess call is followed to
// its effect. A literal path resolves to the exact repository input it names; an
// unresolved value from the environment is a dynamic input that keeps its
// consumer selected without a false name; and a temporary-directory root reaches
// nothing in the repository and confines the package, so fixture roots stay
// distinct from candidate inputs.
func TestRuntimeValueFlowAcceptance(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		source string
		check  func(*testing.T, runtimeInputs)
	}{
		{
			name:   "literal path resolves to its named input",
			source: "package probe\nimport \"os\"\nfunc Value() int { b, _ := os.ReadFile(\"../../docs/config.txt\"); return len(b) }",
			check: func(t *testing.T, in runtimeInputs) {
				if !slices.Contains(in.files, "docs/config.txt") {
					t.Fatalf("resolved literal path did not name its repository input: %+v", in)
				}
			},
		},
		{
			name:   "unresolved path stays a dynamic input",
			source: "package probe\nimport \"os\"\nfunc Value() int { b, _ := os.ReadFile(os.Getenv(\"INPUT\")); return len(b) }",
			check: func(t *testing.T, in runtimeInputs) {
				if in.confined() || len(in.files) != 0 {
					t.Fatalf("unresolved path was confined or given a false name: %+v", in)
				}
			},
		},
		{
			name:   "unresolved command argv follows the subprocess",
			source: "package probe\nimport (\"os\"; \"os/exec\")\nfunc Run() error { return exec.Command(\"go\", \"run\", os.Getenv(\"TARGET\")).Run() }",
			check: func(t *testing.T, in runtimeInputs) {
				if in.confined() {
					t.Fatalf("unresolved command argv was confined: %+v", in)
				}
			},
		},
		{
			name:   "temporary root confines the package",
			source: "package probe\nimport \"os\"\nfunc Value() int { p := os.TempDir(); b, _ := os.ReadFile(p); return len(b) }",
			check: func(t *testing.T, in runtimeInputs) {
				if !in.confined() {
					t.Fatalf("temporary-rooted read escaped confinement: %+v", in)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			dir := filepath.Join(root, "internal", "probe")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "probe.go"), []byte(tc.source), 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := classifyRuntimeInputs(root, dir, []string{"probe.go"})
			if err != nil {
				t.Fatal(err)
			}
			tc.check(t, got)
		})
	}
}
