package main

import (
	"errors"
	"io"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

func TestQwenFourVerificationPublication(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	root := testutil.RepoRoot(t)
	path := filepath.Join(root, "docs/verification/qwen35-4b-retained.json")
	var spec verificationSpecification
	if err := jsonfile.DecodeStrict(path, &spec); err != nil {
		t.Fatal(err)
	}
	// Canonical identity freezes the complete reviewed claim set, not JSON layout.
	const expected = "evidence:sha256:d625011eb8f92c10301bcb71a48bb25e9962277c8920033f38ddc075bbb6c161"
	check := func(spec verificationSpecification) (runrecord.ModelVerification, error) {
		record, err := runrecord.NewModelVerificationCorrection(spec.Model, spec.Name, spec.Claims, spec.Supersedes)
		if err == nil && record.ID.String() != expected {
			err = errors.New("Qwen4 publication changed the reviewed model, source, protocol claims or evidence")
		}
		return record, err
	}
	record, err := check(spec)
	if err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*verificationSpecification){
		func(s *verificationSpecification) { s.Claims = s.Claims[1:] },
		func(s *verificationSpecification) { s.Claims = append(s.Claims, s.Claims[0]) },
		func(s *verificationSpecification) {
			s.Model = testutil.ArtifactID(t, artifact.KindModel, "foreign model")
		},
		func(s *verificationSpecification) { s.Claims[0].Commit = strings.Repeat("b", len(s.Claims[0].Commit)) },
		func(s *verificationSpecification) {
			s.Claims[0].Evidence = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "foreign evidence")}
		},
	} {
		changed := spec
		changed.Claims = slices.Clone(spec.Claims)
		mutate(&changed)
		if _, err := check(changed); err == nil {
			t.Fatal("incomplete, duplicate or changed claim received publication credit")
		}
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, text := range []string{record.ID.String(), "evidence:sha256:8b68cba243001530b04f040b0a9dbd0c0746ba614363db7d00af9093e736560a"} {
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.RequireTypedContent(t.Context(), store, id); err != nil {
			t.Fatal(err)
		}
	}
	refs := []artifact.ID{}
	for _, claim := range record.Claims {
		refs = append(refs, claim.Evidence...)
	}
	if err := artifact.ReadContents(t.Context(), store, refs, func(artifact.Content) error { return nil }); err != nil {
		t.Fatal(err)
	}
	var report modelsReport
	if err := jsonfile.DecodeStrict(filepath.Join(root, modelsReportPath), &report); err != nil {
		t.Fatal(err)
	}
	matches := 0
	for _, prototype := range report.Prototypes {
		for _, model := range prototype.Models {
			if model.Model != record.Model.String() {
				continue
			}
			matches++
			for _, claim := range record.Claims {
				if !slices.ContainsFunc(model.Inference, func(c claimRecord) bool {
					return c.Capability == claim.Capability && c.Tier == string(claim.Tier) && c.Commit == claim.Commit
				}) {
					t.Fatalf("report lost or restamped %s", claim.Capability)
				}
			}
		}
	}
	if matches != 1 {
		t.Fatal("report omitted or duplicated the registered model identity")
	}

	// Tiny fixture exercises the real publisher without copying model or response data.
	fixture := t.TempDir()
	fixtureStore, err := overgodb.Open(fixture)
	if err != nil {
		t.Fatal(err)
	}
	seed := artifact.Batch{Key: "publication-fixture", Artifacts: []artifact.Descriptor{{ID: record.Model}}}
	for _, id := range refs {
		seed.Artifacts = append(seed.Artifacts, artifact.Descriptor{ID: id})
	}
	for _, claim := range record.Claims {
		if claim.Dataset.Valid() {
			seed.Artifacts = append(seed.Artifacts, artifact.Descriptor{ID: claim.Dataset})
		}
	}
	_, seedErr := fixtureStore.Commit(t.Context(), seed)
	if err := errors.Join(seedErr, fixtureStore.Close()); err != nil {
		t.Fatal(err)
	}
	head := func() artifact.CommitID {
		t.Helper()
		store, err := overgodb.OpenReadOnly(fixture)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		id, _ := store.Head()
		return id
	}
	if err := runRecordVerification(path, fixture, io.Discard); err != nil {
		t.Fatal(err)
	}
	first := head()
	if err := runRecordVerification(path, fixture, io.Discard); err != nil {
		t.Fatal(err)
	}
	if head() != first {
		t.Fatal("exact publication replay appended another store transaction")
	}
	if err := artifact.ReadContents(t.Context(), verificationContentFaultReader{Reader: store, id: refs[0], absent: true}, refs, func(artifact.Content) error { return nil }); err == nil {
		t.Fatal("missing claimed evidence gained publication credit")
	}
	t.Log("three exact protocol claims published; original long-form record retained; exact replay adds no transaction or model acquisition")
}
