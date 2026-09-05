package longform

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGuardImpactUsesInferenceOwnership(t *testing.T) {
	t.Run("compiler supplies embedded test-named runtime inputs", func(t *testing.T) {
		repository := t.TempDir()
		for path, data := range map[string]string{
			"go.mod":                            "module overgo\n\ngo 1.26\n",
			"internal/inference/run.go":         "package inference\nimport _ \"embed\"\n//go:embed schema_test.go\nvar schema string\n",
			"internal/inference/schema_test.go": "package inference\n",
			"internal/inference/plain_test.go":  "package inference\n",
			"internal/cuda/executor/run.go":     "package executor\n",
			"internal/modelrecipe/run.go":       "package modelrecipe\n",
		} {
			full := filepath.Join(repository, filepath.FromSlash(path))
			if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(full, []byte(data), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		for _, test := range []struct {
			path string
			want bool
		}{
			{"internal/inference/schema_test.go", true},
			{"internal/inference/plain_test.go", false},
		} {
			affected, reason, err := Affected(t.Context(), repository, []string{test.path})
			if err != nil || affected != test.want || reason == "" {
				t.Fatalf("compiler ownership of %s: %t %s %v", test.path, affected, reason, err)
			}
		}
	})
	root := t.TempDir()
	for _, path := range []string{"internal/runtime/run.go", "internal/runtime/policy.json", "docs/plan.json", "cmd/report/main.go"} {
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte("fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	owners := []string{filepath.Join(root, "internal/runtime")}
	t.Run("embedded test-named input is runtime data", func(t *testing.T) {
		affected, _, err := affectedPaths(root, []string{"internal/runtime/schema_test.go"}, owners, filepath.Join(root, "internal/runtime/schema_test.go"))
		if err != nil || !affected {
			t.Fatalf("embedded runtime input excluded: affected=%t error=%v", affected, err)
		}
	})
	for _, test := range []struct {
		name              string
		paths             []string
		affected, invalid bool
	}{
		{"runtime", []string{"internal/runtime/run.go"}, true, false},
		{"embedded policy", []string{"internal/runtime/policy.json"}, true, false},
		{"kernel", []string{"kernels/cuda/attention.cu"}, true, false},
		{"native asset", []string{"internal/cuda/kernel/attention.ptx"}, true, false},
		{"module", []string{"go.mod"}, true, false},
		{"unknown", nil, true, false},
		{"deleted package", []string{"internal/removed/runtime.go"}, true, false},
		{"tests", []string{"internal/runtime/run_test.go"}, false, false},
		{"host only", []string{"docs/plan.json", "cmd/report/main.go"}, false, false},
		{"Windows path", []string{`internal\runtime\policy.json`}, true, false},
		{"case variant cannot evade ownership", []string{`INTERNAL\RUNTIME\RUN.GO`}, true, false},
		{"traversal", []string{"../runtime.go"}, false, true},
		{"absolute", []string{filepath.Join(root, "internal/runtime/run.go")}, false, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			affected, reason, err := affectedPaths(root, test.paths, owners)
			if (err != nil) != test.invalid || !test.invalid && (affected != test.affected || reason == "") {
				t.Fatalf("affected=%t reason=%q error=%v", affected, reason, err)
			}
		})
	}
}
