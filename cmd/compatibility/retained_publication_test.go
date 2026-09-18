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
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

type retainedPublicationFixture struct {
	Name          string        `json:"name"`
	Specification string        `json:"specification"`
	Record        artifact.ID   `json:"record"`
	Preserve      []artifact.ID `json:"preserve"`
}

func retainedPublicationFixtures(t *testing.T) []retainedPublicationFixture {
	t.Helper()
	var fixtures []retainedPublicationFixture
	if err := jsonfile.DecodeStrict(filepath.Join(testutil.RepoRoot(t), "docs/verification/retained_publications.json"), &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) == 0 {
		t.Fatal("retained publication denominator is empty")
	}
	paths, err := filepath.Glob(filepath.Join(testutil.RepoRoot(t), "docs/verification/*-retained.json"))
	if err != nil {
		t.Fatal(err)
	}
	declared := map[string]bool{}
	for _, path := range paths {
		declared[filepath.Base(path)] = true
	}
	names, records := map[string]bool{}, map[artifact.ID]bool{}
	for _, fixture := range fixtures {
		if fixture.Name == "" || names[fixture.Name] || records[fixture.Record] || fixture.Record.Kind() != artifact.KindEvidence || !filepath.IsLocal(fixture.Specification) || !declared[fixture.Specification] {
			t.Fatal("retained publication fixture is invalid or duplicated")
		}
		delete(declared, fixture.Specification)
		names[fixture.Name], records[fixture.Record] = true, true
		for _, id := range fixture.Preserve {
			if id.Kind() != artifact.KindEvidence {
				t.Fatal("retained predecessor is not evidence")
			}
		}
	}
	if len(declared) != 0 {
		t.Fatal("retained publication specification has no acceptance fixture")
	}
	return fixtures
}

func TestRetainedModelPublications(t *testing.T) {
	for _, fixture := range retainedPublicationFixtures(t) {
		t.Run(fixture.Name, func(t *testing.T) { checkRetainedModelPublication(t, fixture) })
	}
}

func TestQwenFourVerificationPublication(t *testing.T) {
	for _, fixture := range retainedPublicationFixtures(t) {
		if fixture.Name == "qwen-four" {
			if fixture.Record.String() != "evidence:sha256:d625011eb8f92c10301bcb71a48bb25e9962277c8920033f38ddc075bbb6c161" {
				t.Fatal("original Qwen4 acceptance authority changed")
			}
			checkRetainedModelPublication(t, fixture)
			return
		}
	}
	t.Fatal("original Qwen4 acceptance is absent")
}

func checkRetainedModelPublication(t *testing.T, fixture retainedPublicationFixture) {
	t.Helper()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	root := testutil.RepoRoot(t)
	path := filepath.Join(root, "docs/verification", fixture.Specification)
	var spec verificationSpecification
	if err := jsonfile.DecodeStrict(path, &spec); err != nil {
		t.Fatal(err)
	}
	// Canonical identity freezes the complete reviewed claim set, not JSON layout.
	check := func(spec verificationSpecification) (runrecord.ModelVerification, error) {
		record, err := runrecord.NewModelVerificationCorrection(spec.Model, spec.Name, spec.Claims, spec.Supersedes)
		if err == nil && record.ID != fixture.Record {
			err = errors.New("publication changed the reviewed model, source, protocol claims or evidence")
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
	for _, id := range append(slices.Clone(fixture.Preserve), record.ID) {
		if _, err := artifact.RequireTypedContent(t.Context(), store, id); err != nil {
			t.Fatal(err)
		}
	}
	refs := []artifact.ID{}
	for _, claim := range record.Claims {
		refs = append(refs, claim.Evidence...)
	}
	var typed []artifact.ID
	if err := artifact.ReadContents(t.Context(), store, refs, func(content artifact.Content) error {
		if content.Descriptor.Schema == evaluation.EvaluationEvidenceSchema {
			typed = append(typed, content.Descriptor.ID)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	for _, id := range typed {
		if _, err := evaluation.RequireEvaluationEvidence(t.Context(), store, id); err != nil {
			t.Fatal(err)
		}
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
	fixtureRoot := t.TempDir()
	fixtureStore, err := overgodb.Open(fixtureRoot)
	if err != nil {
		t.Fatal(err)
	}
	seed := artifact.Batch{Key: "publication-fixture"}
	seedIDs := append(slices.Clone(refs), record.Model)
	for _, claim := range record.Claims {
		if claim.Dataset.Valid() {
			seedIDs = append(seedIDs, claim.Dataset)
		}
	}
	slices.SortFunc(seedIDs, artifact.CompareID)
	for _, id := range slices.Compact(seedIDs) {
		seed.Artifacts = append(seed.Artifacts, artifact.Descriptor{ID: id})
	}
	for _, id := range record.Supersedes {
		content, err := artifact.RequireTypedContent(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		seed.Contents = append(seed.Contents, content)
	}
	_, seedErr := fixtureStore.Commit(t.Context(), seed)
	if err := errors.Join(seedErr, fixtureStore.Close()); err != nil {
		t.Fatal(err)
	}
	head := func() artifact.CommitID {
		t.Helper()
		store, err := overgodb.OpenReadOnly(fixtureRoot)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		id, _ := store.Head()
		return id
	}
	if err := runRecordVerification(path, fixtureRoot, io.Discard); err != nil {
		t.Fatal(err)
	}
	first := head()
	if err := runRecordVerification(path, fixtureRoot, io.Discard); err != nil {
		t.Fatal(err)
	}
	if head() != first {
		t.Fatal("exact publication replay appended another store transaction")
	}
	if err := artifact.ReadContents(t.Context(), verificationContentFaultReader{Reader: store, id: refs[0], absent: true}, refs, func(artifact.Content) error { return nil }); err == nil {
		t.Fatal("missing claimed evidence gained publication credit")
	}
	t.Logf("%s: %d exact claims and %d predecessors retained; exact replay adds no transaction or model acquisition", spec.Name, len(record.Claims), len(fixture.Preserve))
}
