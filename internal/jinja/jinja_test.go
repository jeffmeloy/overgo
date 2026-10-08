package jinja

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// semanticsScript writes the golden this test reads, from Python's Jinja2 as
// transformers compiles a chat template. It is kept where Python is kept, on
// the machine that ran it and not in git: the golden is the data.
const semanticsScript = "scripts/python_golden_references/jinja_semantics.py"

// TestRendersAsPythonJinjaDoes holds the engine, construct by construct, to
// the language the templates it serves are written in: Python's Jinja2 in
// the environment transformers compiles a chat template in. Each case of
// testdata/semantics.json is one small template over the file's one context,
// read as a request's JSON reaches a template, and the text must be what
// Python renders, or the engine must refuse what Python raises on.
//
// The cases are the ones a chat template leans on: what None and a name that
// is not defined print as and test as, how a float is written, the order a
// mapping is walked in, Python's string methods with and without their
// arguments, tojson with each of its arguments, and how block tags eat the
// lines they stand on.
func TestRendersAsPythonJinjaDoes(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "semantics.json"))
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Context json.RawMessage `json:"context"`
		Cases   []struct {
			Name     string `json:"name"`
			Template string `json:"template"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &document); err != nil || len(document.Cases) == 0 {
		t.Fatalf("no case is on file: %v", err)
	}
	decoded, err := DecodeJSON(document.Context)
	if err != nil {
		t.Fatal(err)
	}
	object, isObject := decoded.(*Dict)
	if !isObject {
		t.Fatal("the context is not an object")
	}
	context := map[string]any{}
	for _, name := range object.keys() {
		context[name], _ = object.get(name)
	}
	raw, err = os.ReadFile(filepath.Join("testdata", "semantics_golden.json"))
	if err != nil {
		t.Fatalf("%v; write the golden with %s", err, semanticsScript)
	}
	var golden struct {
		Renders []struct {
			Name  string `json:"name"`
			Text  string `json:"text"`
			Error string `json:"error"`
		} `json:"renders"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil || len(golden.Renders) != len(document.Cases) {
		t.Fatalf("the golden holds %d renders for %d cases (%v); write it again with %s", len(golden.Renders), len(document.Cases), err, semanticsScript)
	}
	env := New()
	env.SetGlobal("raise_exception", func(args []any, _ map[string]any) (any, error) {
		return nil, errors.New(stringify(args[0]))
	})
	for index, testCase := range document.Cases {
		want := golden.Renders[index]
		if want.Name != testCase.Name {
			t.Fatalf("render %d is of %s and case %d is %s", index, want.Name, index, testCase.Name)
		}
		t.Run(testCase.Name, func(t *testing.T) {
			got, err := env.Render(testCase.Template, context)
			switch {
			case want.Error != "" && err == nil:
				t.Errorf("rendered %q where Python raises %s", got, want.Error)
			case want.Error == "" && err != nil:
				t.Errorf("refused where Python renders %q: %v", want.Text, err)
			case want.Error == "" && got != want.Text:
				t.Errorf("rendered\n%q\nwhere Python renders\n%q", got, want.Text)
			}
		})
	}
}

// TestDecodeJSONRefusesWhatIsNotOneValue holds the reader of a request's JSON
// to refusing what is not one value: nothing, a value cut short, and a value
// with another after it.
func TestDecodeJSONRefusesWhatIsNotOneValue(t *testing.T) {
	for _, text := range []string{``, `{"a": `, `[1, 2`, `{"a": 1} {"b": 2}`, `{1: 2}`} {
		if value, err := DecodeJSON([]byte(text)); err == nil {
			t.Errorf("%q was read as %v", text, value)
		}
	}
}
