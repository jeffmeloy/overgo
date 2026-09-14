package gate

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRuntimeInputBindingsDoNotCrossScopes(t *testing.T) {
	t.Parallel()
	for _, source := range []string{
		"package probe\nimport \"os\"\nfunc Read(path string)([]byte,error){return os.ReadFile(path)}\nfunc Fixture(){path:=os.TempDir();_,_=os.ReadFile(path)}\n",
		"package probe\nimport \"os\"\nconst path=\"fixture\"\nfunc Read(path string)([]byte,error){return os.ReadFile(path)}\n",
		"package probe\nimport \"os\"\nfunc Read(path string)([]byte,error){{path:=os.TempDir();_,_=os.ReadFile(path)};return os.ReadFile(path)}\n",
		"package probe\nimport \"os\"\nfunc Read(path string)([]byte,error){f:=func(){path:=os.TempDir();_,_=os.ReadFile(path)};f();return os.ReadFile(path)}\n",
	} {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, "probe.go"), []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
		inputs, err := classifyRuntimeInputs(root, root, []string{"probe.go"})
		if err != nil {
			t.Fatal(err)
		}
		if inputs.confined() || len(inputs.dynamic) == 0 {
			t.Fatalf("unrelated binding hid caller-controlled read: %s", source)
		}
	}
}
