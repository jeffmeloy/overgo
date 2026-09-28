package inferencesurface

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/model"
)

// TestServingNamesReadServingSourceOnly holds the scan to what selects
// serving code: a string literal in a serving package's source is named with
// its file, once however often it appears there; a test file's literals, and
// the tokenizer's and the projector's, whose names are the formats a model
// file declares beside its architecture, are not.
func TestServingNamesReadServingSourceOnly(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	for name, content := range map[string]string{
		"serving/load.go":      "package serving\n\nfunc special(name string) bool { return name == \"llama\" || name == `llama` }\n",
		"serving/load_test.go": "package serving\n\nconst fixture = \"gemma4\"\n",
		"tokenizer/vocab.go":   "package tokenizer\n\nconst sentencePiece = \"gemma4\"\n",
		"projector/tower.go":   "package projector\n\nconst tower = \"cogvlm\"\n",
	} {
		file := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var packages []Package
	for _, name := range []string{"serving", "tokenizer", "projector"} {
		packages = append(packages, Package{Dir: filepath.Join(root, name), ImportPath: "overgo/internal/" + name})
	}
	named, err := ServingNames(packages)
	if err != nil {
		t.Fatal(err)
	}
	if want := map[string][]string{"llama": {"overgo/internal/serving/load.go"}}; !reflect.DeepEqual(named, want) {
		t.Fatalf("serving names = %v, want %v", named, want)
	}
}

// TestServingCodeNamesNoArchitecture holds the serving code to declarations:
// no string literal in the inference surface's source names a declared
// architecture, so a model is served by what its profile declares and never
// by code its architecture's name selects. A family that needs a behaviour
// declares it in internal/model/architecture_profiles.json.
func TestServingCodeNamesNoArchitecture(t *testing.T) {
	t.Parallel()
	packages, err := Packages(t.Context(), filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	named, err := ServingNames(packages)
	if err != nil {
		t.Fatal(err)
	}
	for _, architecture := range model.SupportedArchitectures() {
		if files := named[architecture]; len(files) != 0 {
			t.Errorf("architecture %s selects serving code in %s; declare what it selects in its profile", architecture, strings.Join(files, ", "))
		}
	}
}
