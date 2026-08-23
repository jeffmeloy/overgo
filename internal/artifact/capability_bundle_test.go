package artifact

import "testing"

func TestCapabilityBundleManifest(t *testing.T) {
	instruction, err := IdentifyBytes(KindFile, []byte("instruction"))
	if err != nil {
		t.Fatal(err)
	}
	resource, err := IdentifyBytes(KindFile, []byte("resource"))
	if err != nil {
		t.Fatal(err)
	}
	bundle, err := NewManifest(KindProfile, []Component{
		{Role: ComponentResource, Name: "schema", Artifact: resource},
		{Role: ComponentInstruction, Name: "system", Artifact: instruction},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateCapabilityBundle(bundle); err != nil || bundle.ID.Kind() != KindProfile ||
		bundle.Components[0].Role != ComponentInstruction {
		t.Fatalf("bundle = %+v, error = %v", bundle, err)
	}
	invalid, err := NewManifest(KindProfile, []Component{{
		Role: ComponentCompanion, Name: "callback", Artifact: resource,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateCapabilityBundle(invalid); err == nil {
		t.Fatal("free-form capability component accepted")
	}
}
