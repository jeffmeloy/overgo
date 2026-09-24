package longform

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// impactFixture writes a module whose surface roots embed a policy and a
// test-named schema, beside a non-embedded note, tests, kernels and host-only
// files; change, when given, rewrites one path after the fixture is written.
func impactFixture(t *testing.T, change string) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod":                            "module fixture\n\ngo 1.26\n",
		"internal/inference/run.go":         "package inference\n\nimport _ \"embed\"\n\n//go:embed policy.json\nvar policy string\n\n//go:embed schema_test.go\nvar schema string\n",
		"internal/inference/policy.json":    `{"heads":2}`,
		"internal/inference/schema_test.go": "package inference\n",
		"internal/inference/plain_test.go":  "package inference\n",
		"internal/inference/notes.json":     `{"note":"not embedded"}`,
		"internal/cuda/executor/run.go":     "package executor\n",
		"internal/modelrecipe/run.go":       "package modelrecipe\n",
		"kernels/manifest.json":             "{}",
		"kernels/cuda/attention.cu":         "// kernel\n",
		"docs/plan.json":                    "{}",
		"cmd/report/main.go":                "package main\n\nfunc main() {}\n",
	}
	if change != "" {
		files[change] += "\n// changed\n"
	}
	for path, data := range files {
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// TestImpactIsTheSurface holds scheduling and evidence identity to one
// dependency description: for every path, impact selection calls the change
// affected exactly when it moves the surface digest that keys retained
// evidence -- an embedded file, a test-named file the package embeds, a
// kernel source or a closure source does; a non-embedded note, an ordinary
// test or a host-only file does not. Any spelling of a path is judged alike,
// and a path outside the repository is refused. go.mod is the one
// conservative case: a path cannot say whether a version moved, so it is
// always measured.
func TestImpactIsTheSurface(t *testing.T) {
	t.Parallel()
	base, err := Surface(t.Context(), impactFixture(t, ""))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{
		"internal/inference/run.go", "internal/inference/policy.json", "internal/inference/schema_test.go",
		"internal/inference/plain_test.go", "internal/inference/notes.json", "internal/cuda/executor/run.go",
		"kernels/cuda/attention.cu", "kernels/manifest.json", "docs/plan.json", "cmd/report/main.go",
	} {
		root := impactFixture(t, path)
		moved, err := Surface(t.Context(), root)
		if err != nil {
			t.Fatal(err)
		}
		affected, reason, err := Affected(t.Context(), root, []string{path})
		if err != nil || affected != (moved != base) || reason == "" {
			t.Fatalf("%s: affected=%v but the digest moved=%v (%s, %v)", path, affected, moved != base, reason, err)
		}
	}
	root := impactFixture(t, "")
	spellings := []string{`internal\inference\policy.json`}
	if runtime.GOOS == "windows" {
		spellings = append(spellings, `INTERNAL\INFERENCE\RUN.GO`)
	}
	for _, spelling := range append(spellings, "go.mod") {
		if affected, _, err := Affected(t.Context(), root, []string{spelling}); err != nil || !affected {
			t.Fatalf("%s: affected=%v %v", spelling, affected, err)
		}
	}
	if affected, _, err := Affected(t.Context(), root, nil); err != nil || !affected {
		t.Fatalf("unknown scope was not measured: %v %v", affected, err)
	}
	for _, outside := range []string{"../runtime.go", filepath.Join(root, "internal/inference/run.go")} {
		if _, _, err := Affected(t.Context(), root, []string{outside}); err == nil {
			t.Fatalf("%s was accepted", outside)
		}
	}
}
