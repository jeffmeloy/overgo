package codemanifest

import (
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

func TestDeltaDetectsAddedAndRemovedSymbols(t *testing.T) {
	base := completeFixtureManifest(t)
	candidateValue := fixtureManifest()
	candidateValue.Uncertainty = nil
	candidateValue.Symbols = candidateValue.Symbols[1:]
	candidateValue.References = nil
	candidate, err := codec.New(candidateValue)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := Diff(base, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if len(delta.Symbols) != 1 || delta.Symbols[0].Kind != ChangeRemoved {
		t.Fatalf("symbol delta = %+v", delta.Symbols)
	}
}

func TestSignatureAndBodyChangesRemainDistinct(t *testing.T) {
	root := t.TempDir()
	name := "internal/example/run.go"
	selection := gosource.BuildSelection{
		Context: "linux/amd64", Root: root, Files: map[string]bool{name: true},
		Packages: map[string]string{name: "overgo/internal/example"},
	}
	writeGeneratorFixture(t, root, name, "package example\nfunc Run() int { return 1 }\n")
	base := generateForDelta(t, root, selection)
	writeGeneratorFixture(t, root, name, "package example\nfunc Run() int { return 2 }\n")
	bodyCandidate := generateForDelta(t, root, selection)
	bodyDelta, err := Diff(base, bodyCandidate)
	if err != nil {
		t.Fatal(err)
	}
	if len(bodyDelta.Symbols) != 1 || bodyDelta.Symbols[0].SignatureChanged || !bodyDelta.Symbols[0].BodyChanged {
		t.Fatalf("body delta = %+v", bodyDelta.Symbols)
	}
	writeGeneratorFixture(t, root, name, "package example\nfunc Run(value int) int { return 2 }\n")
	signatureCandidate := generateForDelta(t, root, selection)
	signatureDelta, err := Diff(bodyCandidate, signatureCandidate)
	if err != nil {
		t.Fatal(err)
	}
	if len(signatureDelta.Symbols) != 1 || !signatureDelta.Symbols[0].SignatureChanged {
		t.Fatalf("signature delta = %+v", signatureDelta.Symbols)
	}
}

func TestExternalInputDelta(t *testing.T) {
	baseValue := fixtureManifest()
	baseValue.Uncertainty = nil
	base, err := codec.New(baseValue)
	if err != nil {
		t.Fatal(err)
	}
	candidateValue := baseValue
	candidateValue.ExternalInputs = []ExternalInput{{
		Path: "kernels/manifest.json", Kind: "kernel-manifest", Owner: "internal/cuda/kernel",
		ContentID: "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	}}
	candidate, err := codec.New(candidateValue)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := Diff(base, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if len(delta.ExternalInputs) != 1 || delta.ExternalInputs[0].Kind != ChangeModified {
		t.Fatalf("external delta = %+v", delta.ExternalInputs)
	}
}

func completeFixtureManifest(t *testing.T) Manifest {
	t.Helper()
	value := fixtureManifest()
	value.Uncertainty = nil
	manifest, err := codec.New(value)
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}

func generateForDelta(t *testing.T, root string, selection gosource.BuildSelection) Manifest {
	t.Helper()
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := Generate(snapshot, []gosource.BuildSelection{selection}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}
