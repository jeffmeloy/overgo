package artifact

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestCapabilityBundleManifest(t *testing.T) {
	instruction, err := IdentifyBytes(KindFile, []byte("instruction"))
	testcheck.NoError(t, err)
	resource, err := IdentifyBytes(KindFile, []byte("resource"))
	testcheck.NoError(t, err)
	bundle, err := NewManifest(KindProfile, []Component{
		{Role: ComponentResource, Name: "schema", Artifact: resource},
		{Role: ComponentInstruction, Name: "system", Artifact: instruction},
	})
	testcheck.NoError(t, err)
	if err := ValidateCapabilityBundle(bundle); err != nil || bundle.ID.Kind() != KindProfile ||
		bundle.Components[0].Role != ComponentInstruction {
		t.Fatalf("bundle = %+v, error = %v", bundle, err)
	}
	invalid, err := NewManifest(KindProfile, []Component{{
		Role: ComponentCompanion, Name: "callback", Artifact: resource,
	}})
	testcheck.NoError(t, err)
	if err := ValidateCapabilityBundle(invalid); err == nil {
		t.Fatal("free-form capability component accepted")
	}
}
