//go:build windows

package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestObjectControlsAreTakenFieldByField holds the settings a request holds as objects and lists
// (moss-ultra-compare/object-controls): MOSS's page declares a MOSS Ultra section's schedule with its samplers and
// schedulers as choices, its canvas, its sharpening, the adapters it stacks with the model's adapters as choices, its
// depth control with its map a slot of image media, and its layout with its regions and palette as lists; a value is
// taken field by field and item by item, and refused where a part is undeclared, mistyped or missing, or a list is not
// one; a flow profile's page declares none of them; and a capability declaring an object with no fields, or a list
// with no item, is refused.
func TestObjectControlsAreTakenFieldByField(t *testing.T) {
	t.Parallel()
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "adapters"), 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "adapters", "moss-control.safetensors"), nil, 0o600))
	catalog := BindGenerationCatalog(map[recipe.Task]mediacapability.Capability{}, mediacapability.Controls, mediacapability.OutputContent)
	controls, refusal := catalog.Controls(modelrecipe.ModuleLatentImagePrepare, root)
	if refusal != "" {
		t.Fatal(refusal)
	}
	testcheck.NoError(t, validateWorkflowControls(controls))
	held := map[string]workflowControl{}
	for _, control := range controls {
		held[control.Name] = control
	}
	field := func(control workflowControl, name string) workflowControl {
		t.Helper()
		at := slices.IndexFunc(control.Fields, func(c workflowControl) bool { return c.Name == name })
		if at < 0 {
			t.Fatalf("%s declares no field %s: %+v", control.Name, name, control.Fields)
		}
		return control.Fields[at]
	}
	for _, name := range []string{"schedule", "canvas", "sharpen", "control", "layout"} {
		if held[name].Type != workflowControlObject || held[name].Required {
			t.Errorf("MOSS's page declares %s as %+v; want an optional object", name, held[name])
		}
	}
	if sampler, scheduler := field(held["schedule"], "sampler"), field(held["schedule"], "name"); !slices.Equal(sampler.Choices, []string{"euler", "er_sde"}) || !slices.Equal(scheduler.Choices, []string{"bong_tangent", "beta57"}) {
		t.Errorf("the schedule offers the samplers %v and the schedulers %v", sampler.Choices, scheduler.Choices)
	}
	for _, name := range []string{"aspect", "megapixels", "multiple"} {
		field(held["canvas"], name)
	}
	for _, name := range []string{"radius", "sigma", "alpha"} {
		field(held["sharpen"], name)
	}
	if depth := field(held["control"], "map"); depth.Type != workflowControlArtifact || depth.Media != "image" || !depth.Required {
		t.Errorf("the depth control's map is %+v; want a required slot of image media", depth)
	}
	adapters := held["adapters"]
	if adapters.Type != workflowControlList || adapters.Item == nil || adapters.Item.Type != workflowControlObject || !slices.Equal(field(*adapters.Item, "name").Choices, []string{"moss-control"}) {
		t.Errorf("the adapters are %+v; want a list of objects naming the model's adapters", adapters)
	}
	regions, palette := field(held["layout"], "regions"), field(held["layout"], "palette")
	if regions.Type != workflowControlList || regions.Item == nil || len(regions.Item.Fields) == 0 || palette.Type != workflowControlList || palette.Item == nil || palette.Item.Type != workflowControlText {
		t.Errorf("the layout's regions are %+v and its palette %+v; want a list of objects and a list of text", regions, palette)
	}

	picture := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, []byte("a depth map")))(t)
	taken := `{"prompt":"a fox","schedule":{"sampler":"er_sde","name":"bong_tangent"},"canvas":null,"adapters":[{"name":"moss-control","strength":0.5},{"name":"moss-control"}],` +
		`"control":{"map":"` + picture.String() + `","strength":1,"end":0.5},"layout":{"regions":[{"x":1,"y":2,"w":3,"h":4,"text":"OVERGO"}],"palette":["#ffffff"]}}`
	testcheck.NoError(t, validateWorkflowInput(controls, json.RawMessage(taken)))
	for name, refused := range map[string]string{
		"an undeclared field":   `{"prompt":"a fox","schedule":{"samplr":"er_sde"}}`,
		"a mistyped field":      `{"prompt":"a fox","canvas":{"megapixels":"many"}}`,
		"a missing map":         `{"prompt":"a fox","control":{"strength":1}}`,
		"a map that is no file": `{"prompt":"a fox","control":{"map":"a picture"}}`,
		"a mistyped item":       `{"prompt":"a fox","adapters":[{"name":3}]}`,
		"an object for a list":  `{"prompt":"a fox","adapters":{"name":"moss-control"}}`,
		"a mistyped region":     `{"prompt":"a fox","layout":{"regions":[{"x":"left"}]}}`,
	} {
		if validateWorkflowInput(controls, json.RawMessage(refused)) == nil {
			t.Errorf("%s was taken: %s", name, refused)
		}
	}

	krea := t.TempDir()
	testutil.WriteImageProfileFixture(t, krea)
	flow, refusal := catalog.Controls(modelrecipe.ModuleLatentImagePrepare, krea)
	if refusal != "" {
		t.Fatal(refusal)
	}
	if slices.ContainsFunc(flow, func(c workflowControl) bool { return c.Type == workflowControlObject || c.Type == workflowControlList }) {
		t.Errorf("a flow profile's page declares a setting it does not take: %+v", flow)
	}
	for name, declared := range map[string]workflowControl{
		"an object with no fields": {Name: "schedule", Type: workflowControlObject},
		"a list with no item":      {Name: "adapters", Type: workflowControlList},
		"a scalar with fields":     {Name: "steps", Type: workflowControlInteger, Fields: []workflowControl{{Name: "x", Type: workflowControlText}}},
		"an unnamed field":         {Name: "canvas", Type: workflowControlObject, Fields: []workflowControl{{Type: workflowControlText}}},
	} {
		if validateWorkflowControls([]workflowControl{declared}) == nil {
			t.Errorf("%s was declared", name)
		}
	}
}
