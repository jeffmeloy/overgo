package modelintake

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

func TestDeterministicImportAcrossFamilies(t *testing.T) {
	root := t.TempDir()
	testutil.WriteImageInventoryFixture(t, root)
	inspection, err := InspectImport(t.Context(), root, "")
	if err != nil {
		t.Fatal(err)
	}
	var description importDescription
	if err := json.Unmarshal(inspection.Content.Data, &description); err != nil {
		t.Fatal(err)
	}
	if description.Qualification != "structural-only" || len(description.Components) != 5 || len(inspection.Inventory.TensorInventory.Tensors) != 4 {
		t.Fatal("incomplete structural description", description)
	}
	for _, component := range description.Components {
		if component.Unsupported == "" || len(component.Operations) != 0 || len(component.Bindings) != 0 || len(component.Policies) == 0 {
			t.Fatal("inventoried component gained an unaudited mapping", component)
		}
		switch component.Declaration.Name {
		case "text_encoder":
			if component.Declaration.Class != "Qwen3VLModel" || !strings.Contains(component.Unsupported, "qwen3_vl") {
				t.Fatal("conditioner architecture lost", component)
			}
		case "vae":
			if component.Declaration.Class != "AutoencoderKLQwenImage" {
				t.Fatal("decoder architecture lost", component)
			}
		case "scheduler":
			if string(component.Policies[0].Parameters["use_dynamic_shifting"]) != "true" {
				t.Fatal("scheduler policy lost", component)
			}
		}
	}
	gguf, err := InspectImport(t.Context(), testutil.HermeticLlamaGGUF(t, 16), "")
	if err != nil {
		t.Fatal(err)
	}
	description = importDescription{}
	if err := json.Unmarshal(gguf.Content.Data, &description); err != nil {
		t.Fatal(err)
	}
	if len(description.Components) != 1 || description.Components[0].Unsupported != "" || len(description.Components[0].Operations) == 0 {
		t.Fatal("registered GGUF mapping lost", description)
	}
}
