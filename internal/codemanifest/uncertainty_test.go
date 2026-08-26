package codemanifest

import "testing"

func TestUncertaintyKindsAreClosed(t *testing.T) {
	value := fixtureManifest()
	value.Uncertainty[0].Kind = UncertaintyKind("invented")
	if _, err := codec.New(value); err == nil {
		t.Fatal("unknown uncertainty kind was accepted")
	}
}

func TestUnknownCannotExclude(t *testing.T) {
	uncertain, err := codec.New(fixtureManifest())
	if err != nil {
		t.Fatal(err)
	}
	if err := uncertain.ExclusionAuthority(); err == nil {
		t.Fatal("manifest with uncertainty authorized exclusions")
	}

	completeValue := fixtureManifest()
	completeValue.Uncertainty = nil
	complete, err := codec.New(completeValue)
	if err != nil {
		t.Fatal(err)
	}
	if err := complete.ExclusionAuthority(); err != nil {
		t.Fatalf("complete manifest did not authorize exclusions: %v", err)
	}

	complete.SourceIdentity = fixtureDigest[:len(fixtureDigest)-1]
	if err := complete.ExclusionAuthority(); err == nil {
		t.Fatal("invalid manifest authorized exclusions")
	}
}
