package clioptions

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestCLIInferenceDependencyBoundary(t *testing.T) {
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("source location unavailable")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(source), "..", ".."))
	const inference = "overgo/internal/inference"
	dependencies := func(overlay string, targets ...string) []string {
		t.Helper()
		args := []string{"list", "-deps", "-f", "{{.ImportPath}}"}
		if overlay != "" {
			args = append(args, "-overlay", overlay)
		}
		output, err := CombinedOutputIn(root, nil, "go", append(args, targets...)...)
		if err != nil {
			t.Fatalf("dependency listing: %v\n%s", err, output)
		}
		return strings.Fields(output)
	}
	for _, target := range []string{"./internal/clioptions", "./cmd/plan", "./cmd/cuda-info"} {
		got := dependencies("", target)
		if slices.Contains(got, inference) {
			t.Fatalf("generic consumer %s still imports inference", target)
		}
		if target == "./internal/clioptions" && slices.Contains(got, "overgo/internal/cuda/driver") {
			t.Fatal("generic CLI helpers import the CUDA driver")
		}
	}
	if got := dependencies("", "./internal/modelcli"); !slices.Contains(got, inference) {
		t.Fatal("model opener lost its inference implementation")
	}
	// An overlay seeds the former coupling without changing the checkout or
	// compiling a model. The same graph check must observe its transitive effect.
	original := filepath.Join(root, "internal", "clioptions", "command.go")
	data, err := os.ReadFile(original)
	if err != nil {
		t.Fatal(err)
	}
	replacement := strings.Replace(string(data), "package clioptions", "package clioptions\nimport _ \""+inference+"\"", 1)
	probe := filepath.Join(t.TempDir(), "command.go")
	if err := os.WriteFile(probe, []byte(replacement), PrivateFileMode); err != nil {
		t.Fatal(err)
	}
	overlay := filepath.Join(t.TempDir(), "overlay.json")
	encoded, err := json.Marshal(struct{ Replace map[string]string }{map[string]string{original: probe}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(overlay, encoded, PrivateFileMode); err != nil {
		t.Fatal(err)
	}
	if got := dependencies(overlay, "./cmd/plan"); !slices.Contains(got, inference) {
		t.Fatal("seeded inference coupling escaped the dependency assertion")
	}
}
