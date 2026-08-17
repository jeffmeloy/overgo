package codeprofile

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func TestASTProfileEvidence(t *testing.T) {
	gate, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("gate"))
	if err != nil {
		t.Fatal(err)
	}
	profile := Profile{
		Production:           Partition{Files: 2, Nodes: 20},
		Functions:            []Function{{File: "internal/p.go", Name: "f", Nodes: 5}},
		Clones:               []Clone{{Fingerprint: strings.Repeat("0", sha256HexLength), Nodes: 5, Functions: []string{"a:f", "b:g"}}},
		DuplicateExcessNodes: 5,
	}
	evidence, err := NewEvidence("0123456789abcdef0123456789abcdef01234567", gate, profile)
	if err != nil {
		t.Fatal(err)
	}
	content, err := evidence.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := evidenceCodec.Parse(content.Data)
	if err != nil || parsed.ID != evidence.ID || len(parsed.Lineage()) != 1 {
		t.Fatalf("profile evidence = (%+v, %v)", parsed, err)
	}
	parsed.Profile.DuplicateExcessNodes++
	if err := parsed.ValidateIdentity(); err == nil {
		t.Fatal("mutated profile evidence retained its identity")
	}
}
