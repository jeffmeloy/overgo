//go:build windows

package mediacapability

import (
	"encoding/json"
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestMOSSImageGenerationContract(t *testing.T) {
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	controls, refusal := Controls(modelrecipe.ModuleLatentImagePrepare, root)
	if refusal != "" {
		t.Fatal(refusal)
	}
	held := map[string]Control{}
	for _, c := range controls {
		held[c.Name] = c
	}
	for _, name := range []string{"prompt", "prompt_mode", "negative_prompt", "width", "height", "steps", "seed", "guidance_scale"} {
		if _, ok := held[name]; !ok {
			t.Fatal("missing supported control", name)
		}
	}
	if len(held) != 8 || held["guidance_scale"].Type != ControlNumber || len(held["prompt_mode"].Choices) != 2 || held["prompt_mode"].Help == "" {
		t.Fatal("unsupported controls", held)
	}
	for _, name := range []string{"negative_prompt", "steps", "seed", "guidance_scale"} {
		if !held[name].Secondary() || held[name].Required {
			t.Fatal("optional tuning control is not secondary", name)
		}
	}
	for name, want := range map[string]Bounds{"width": {Default: 1024, Step: 16}, "height": {Default: 1024, Step: 16}, "steps": {Default: 50, Step: 1}, "guidance_scale": {Default: 3}} {
		if held[name].Bounds == nil || *held[name].Bounds != want {
			t.Fatal("source default differs", name, held[name])
		}
	}
	raw, err := Catalog[recipe.TaskImageGen].ValidationInput(root, "  blue cube\n")
	if err != nil {
		t.Fatal(err)
	}
	var input latentimage.Request
	if err = json.Unmarshal([]byte(raw), &input); err != nil {
		t.Fatal(err)
	}
	if input.Prompt != "  blue cube\n" || input.Width != 16 || input.Height != 16 || input.Steps != 1 || input.GuidanceScale != nil {
		t.Fatal("source-derived validation request differs", input)
	}
	krea := t.TempDir()
	testutil.WriteImageProfileFixture(t, krea)
	controls, refusal = Controls(modelrecipe.ModuleLatentImagePrepare, krea)
	if refusal != "" {
		t.Fatal(refusal)
	}
	for _, c := range controls {
		if c.Name == "negative_prompt" || c.Name == "guidance_scale" || c.Name == "prompt_mode" {
			t.Fatal("unsupported override offered for another profile", c)
		}
	}
}
