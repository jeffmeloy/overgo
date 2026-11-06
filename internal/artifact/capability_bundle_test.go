package artifact

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestCapabilityBundleManifest(t *testing.T) {
	instruction := testcheck.Must(IdentifyBytes(KindFile, []byte("instruction")))(t)
	resource := testcheck.Must(IdentifyBytes(KindFile, []byte("resource")))(t)
	bundle := testcheck.Must(NewManifest(KindProfile, []Component{
		{Role: ComponentResource, Name: "schema", Artifact: resource},
		{Role: ComponentInstruction, Name: "system", Artifact: instruction},
	}))(t)
	if err := ValidateCapabilityBundle(bundle); err != nil || bundle.ID.Kind() != KindProfile ||
		bundle.Components[0].Role != ComponentInstruction {
		t.Fatalf("bundle = %+v, error = %v", bundle, err)
	}
	invalid := testcheck.Must(NewManifest(KindProfile, []Component{{
		Role: ComponentCompanion, Name: "callback", Artifact: resource,
	}}))(t)
	if err := ValidateCapabilityBundle(invalid); err == nil {
		t.Fatal("free-form capability component accepted")
	}
}
