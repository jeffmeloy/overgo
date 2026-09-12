package main

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

const (
	miniNativeModel  = "model:sha256:3007c05b8ece556726a37980069cf6c0f1f966a48572b1c130c5643810c23a32"
	miniNativeRecipe = "recipe:sha256:5155b720ce15e6b05acf713a5f9c5bf86bf7e52963178c18503b03206578543c"
)

type miniNativeCase struct {
	Name      string  `json:"name"`
	Prompt    string  `json:"prompt"`
	MaxTokens int     `json:"max_tokens"`
	Chat      bool    `json:"chat"`
	Input     []int32 `json:"input"`
	Output    []int32 `json:"output"`
	Text      string  `json:"text"`
}

type miniNativeCapture struct {
	Cases []miniNativeCase `json:"cases"`
}

func checkMiniNativeCaptures(goCapture, native miniNativeCapture) error {
	names := []string{"capital", "count", "code", "integer-addition"}
	if len(goCapture.Cases) != len(names) || len(native.Cases) != len(names) {
		return fmt.Errorf("MiniCPM native reference: incomplete case denominator")
	}
	inputs, outputs := 0, 0
	for i, name := range names {
		a, b := goCapture.Cases[i], native.Cases[i]
		if a.Name != name || b.Name != name || a.Prompt == "" || len(a.Input) == 0 || len(a.Output) == 0 || len(a.Output) > a.MaxTokens ||
			!slices.Equal(a.Input, b.Input) || !slices.Equal(a.Output, b.Output) || a.Text != b.Text {
			return fmt.Errorf("MiniCPM native reference: case %s differs", name)
		}
		inputs += len(a.Input)
		outputs += len(a.Output)
	}
	if inputs != 48 || outputs != 82 {
		return fmt.Errorf("MiniCPM native reference: token denominator differs")
	}
	return nil
}

func readMiniNativeCaptures(t testing.TB) {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id, err := artifact.ParseID("evidence:sha256:39ce1caf0a126eb35fb73510350972a27d6acf8a483e326e80b80425b1311d5a")
	if err != nil {
		t.Fatal(err)
	}
	root, err := artifact.RequireTypedContent(t.Context(), store, id)
	if err != nil {
		t.Fatal(err)
	}
	var provenance struct {
		Schema       string                 `json:"schema"`
		Model        string                 `json:"model"`
		Recipe       string                 `json:"recipe"`
		GoCommit     string                 `json:"go_commit"`
		NativeCommit string                 `json:"native_commit"`
		Inputs       map[string]artifact.ID `json:"inputs"`
	}
	if err := json.Unmarshal(root.Data, &provenance); err != nil {
		t.Fatal(err)
	}
	if provenance.Schema != "overgo/native-reference-provenance/v1" ||
		provenance.Model != miniNativeModel ||
		provenance.Recipe != miniNativeRecipe ||
		provenance.GoCommit != "f278ca230121df1d0bedf8fc58bf7881a9ebc475" || provenance.NativeCommit != "42fc243060709331ff9b158a9ed2cbe37219ae83" {
		t.Fatal("MiniCPM native reference: producer or model authority differs")
	}
	var goCapture, native miniNativeCapture
	for _, name := range []string{"go_capture", "native_capture", "native_input", "go_source", "native_source", "native_build", "conversion_source", "conversion_verifier", "conversion_verification"} {
		content, err := artifact.RequireTypedContent(t.Context(), store, provenance.Inputs[name])
		if err != nil {
			t.Fatal(err)
		}
		switch name {
		case "go_capture":
			err = json.Unmarshal(content.Data, &goCapture)
		case "native_capture":
			err = json.Unmarshal(content.Data, &native)
		case "conversion_verification":
			var conversion struct {
				SourceSHA256, DestinationSHA256      string
				WidenedVectors, ByteIdenticalTensors int
			}
			err = json.Unmarshal(content.Data, &conversion)
			if err == nil && (conversion.SourceSHA256 != "55f23e065a62cf5a831359aa9ef7856cbb27bf8dee8439dcfe351a7e9be2eb8f" || conversion.DestinationSHA256 != "9a25e3e5f642f2b18e7500e940e1ef0cecf53c811662f102983b8531d211b2ce" || conversion.WidenedVectors != 49 || conversion.ByteIdenticalTensors != 170) {
				t.Fatal("MiniCPM native reference: value-preserving conversion differs")
			}
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := checkMiniNativeCaptures(goCapture, native); err != nil {
		t.Fatal(err)
	}
	deviceID, err := artifact.ParseID("evidence:sha256:2b3944a01a6fad3a8db1c5742aea84ea95ffccccd0929c4730b15196f33ef7d9")
	if err != nil {
		t.Fatal(err)
	}
	deviceRoot, err := artifact.RequireTypedContent(t.Context(), store, deviceID)
	if err != nil {
		t.Fatal(err)
	}
	var deviceBinding struct {
		Schema          string                 `json:"schema"`
		NativeReference artifact.ID            `json:"native_reference"`
		Producer        string                 `json:"producer"`
		Inputs          map[string]artifact.ID `json:"inputs"`
	}
	if err := json.Unmarshal(deviceRoot.Data, &deviceBinding); err != nil {
		t.Fatal(err)
	}
	if deviceBinding.Schema != "overgo/native-device-reference/v1" || deviceBinding.NativeReference != id || deviceBinding.Producer != provenance.GoCommit {
		t.Fatal("MiniCPM device reference: native or producer authority differs")
	}
	if _, err := artifact.RequireTypedContent(t.Context(), store, deviceBinding.Inputs["source"]); err != nil {
		t.Fatal(err)
	}
	deviceContent, err := artifact.RequireTypedContent(t.Context(), store, deviceBinding.Inputs["capture"])
	if err != nil {
		t.Fatal(err)
	}
	var device struct {
		miniNativeCapture
		Identity    modelrecipe.ProgramIdentity `json:"identity"`
		Environment runrecord.Environment       `json:"environment"`
		Source      string                      `json:"source"`
		Sampling    string                      `json:"sampling"`
	}
	if err := json.Unmarshal(deviceContent.Data, &device); err != nil {
		t.Fatal(err)
	}
	if device.Identity.Model.String() != miniNativeModel || device.Identity.Recipe.String() != miniNativeRecipe || device.Source != provenance.GoCommit || device.Sampling != "device-greedy" || device.Environment.Backend != "cuda" {
		t.Fatal("MiniCPM device reference: executed model, recipe or protocol differs")
	}
	if _, err := runrecord.NewEnvironment(device.Environment); err != nil {
		t.Fatal(err)
	}
	definition, err := recipe.RequireDefinition(t.Context(), store, device.Identity.Recipe)
	if err != nil {
		t.Fatal(err)
	}
	for role, expected := range map[recipe.DependencyRole]artifact.ID{
		recipe.DependencyModel:      device.Identity.Model,
		recipe.DependencyProfile:    device.Identity.Profile,
		recipe.DependencyDefinition: device.Identity.Definition,
	} {
		if actual, found := definition.PrimaryDependency(role); !found || actual != expected {
			t.Fatalf("MiniCPM device reference: recipe %s binding differs", role)
		}
	}
	if err := checkMiniNativeCaptures(device.miniNativeCapture, native); err != nil {
		t.Fatal(err)
	}
}

func TestAcceptedMiniCPMNativeReference(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT for retained MiniCPM native evidence")
	}
	readMiniNativeCaptures(t)
	t.Log("four host-greedy and four device-greedy cases match pinned native CPU: exact 48 input and 82 output tokens per path; source, loaded model/recipe and conversion bound; model executions=0")
}

func TestMiniCPMNativeContract(t *testing.T) {
	for _, mutation := range []string{"valid", "missing", "duplicate", "input", "output", "text", "budget", "denominator"} {
		t.Run(mutation, func(t *testing.T) {
			var a, b miniNativeCapture
			for i, name := range []string{"capital", "count", "code", "integer-addition"} {
				row := miniNativeCase{Name: name, Prompt: name, MaxTokens: []int{24, 24, 32, 16}[i], Input: make([]int32, []int{6, 9, 6, 27}[i]), Output: make([]int32, []int{24, 24, 32, 2}[i]), Text: name}
				a.Cases = append(a.Cases, row)
				row.Input, row.Output = slices.Clone(row.Input), slices.Clone(row.Output)
				b.Cases = append(b.Cases, row)
			}
			switch mutation {
			case "missing":
				b.Cases = b.Cases[:3]
			case "duplicate":
				b.Cases[1] = b.Cases[0]
			case "input":
				b.Cases[0].Input[0]++
			case "output":
				b.Cases[0].Output[0]++
			case "text":
				b.Cases[0].Text += "changed"
			case "budget":
				a.Cases[0].MaxTokens--
			case "denominator":
				a.Cases[0].Input = append(a.Cases[0].Input, 0)
				b.Cases[0].Input = append(b.Cases[0].Input, 0)
			}
			err := checkMiniNativeCaptures(a, b)
			if (err == nil) != (mutation == "valid") {
				t.Fatalf("acceptance error = %v", err)
			}
		})
	}
}
