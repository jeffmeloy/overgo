package gate

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRuntimeInputsIndirectAndConditional(t *testing.T) {
	t.Parallel()
	for name, source := range map[string]string{
		"function_alias": `package probe
 import "os"
 func Read() ([]byte,error) { read := os.ReadFile; return read(os.Getenv("INPUT")) }`,
		"go_flags": `package probe
 import("os"; "os/exec")
 func Run() error { return exec.Command("go", "run", "-tags=modeltest", os.Getenv("TARGET")).Run() }`,
		"branch_merge": `package probe
 import "os"
 //overgo:runtime-inputs caller
 func Read(input string, choose bool) ([]byte,error) { p := os.Getenv("INPUT"); if choose { p = input }; return os.ReadFile(p) }`,
	} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			dir := filepath.Join(root, "internal", "probe")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "probe.go"), []byte(source), 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := classifyRuntimeInputs(root, dir, []string{"probe.go"})
			if err != nil {
				t.Fatal(err)
			}
			if got.confined() {
				t.Fatalf("unresolved runtime input incorrectly confined: %+v", got)
			}
		})
	}
}
