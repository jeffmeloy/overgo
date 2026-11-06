package agenttool

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func TestCandidateCatalogStagesVerifiesAndActivatesExactly(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	compilation := testcheck.Must(CompileOpenAPI(strings.NewReader(openAPIFixture), "https://api.example.test/v1"))(t)
	source := testcheck.Must(compilation.SourceContent())(t)
	staging := testcheck.Must(StageManualCandidates(ctx, store, source, compilation.Manuals))(t)
	if _, found, err := store.ResolveAlias(ctx, RegisteredAlias("weather.forecast")); err != nil || found {
		t.Fatalf("candidate changed a registered alias: found=%t err=%v", found, err)
	}
	if _, err := ActivateCandidateCatalog(ctx, store, staging.Candidate.ID, staging.Candidate.ID); err == nil {
		t.Fatal("candidate activated without verification evidence")
	}
	verification, _, err := PublishCandidateVerification(ctx, store, staging.Candidate.ID, NewOperatorExecutor())
	testcheck.NoError(t, err)
	activation := testcheck.Must(ActivateCandidateCatalog(ctx, store, staging.Candidate.ID, verification.ID))(t)
	if !activation.Changed || !activation.Coverage.Complete || activation.Snapshot != staging.Snapshot.ID {
		t.Fatalf("activation = %+v", activation)
	}
	manual, err := ResolveRegisteredManual(ctx, store, "weather.forecast")
	if err != nil || manual.ID != compilation.Manuals[0].ID {
		t.Fatalf("resolved manual = %+v, %v", manual, err)
	}
}

func TestCandidateGrantRejectsDrift(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	compilation := testcheck.Must(CompileOpenAPI(strings.NewReader(openAPIFixture), "https://api.example.test/v1"))(t)
	source := testcheck.Must(compilation.SourceContent())(t)
	staging := testcheck.Must(StageManualCandidates(ctx, store, source, compilation.Manuals))(t)
	missingHTTP := NewOperatorExecutor()
	delete(missingHTTP.adapters, TransportHTTP)
	if _, _, err := PublishCandidateVerification(ctx, store, staging.Candidate.ID, missingHTTP); err == nil {
		t.Fatal("candidate received a grant without its host transport capability")
	}
	grant, _, err := PublishCandidateVerification(ctx, store, staging.Candidate.ID, NewOperatorExecutor())
	testcheck.NoError(t, err)
	foreign := testcheck.Must(artifact.JSONID(artifact.KindProfile, "drift"))(t)
	mutations := map[string]func(*CandidateVerification){
		"tool set":        func(value *CandidateVerification) { value.ToolSet = foreign },
		"manual schemas":  func(value *CandidateVerification) { value.Schemas = foreign },
		"endpoint scopes": func(value *CandidateVerification) { value.Endpoints = foreign },
		"host capability": func(value *CandidateVerification) { value.Host = foreign },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			drifted := grant
			mutate(&drifted)
			drifted = testcheck.Must(candidateVerificationCodec.New(drifted))(t)
			content := testcheck.Must(candidateVerificationCodec.Content(drifted))(t)
			batch := testcheck.Must(artifact.NewDocumentBatch("candidate/grant-drift/"+name, []artifact.Content{content}, nil, nil))(t)
			if _, err := store.Commit(ctx, batch); err != nil {
				t.Fatal(err)
			}
			if _, err := ActivateCandidateCatalog(ctx, store, staging.Candidate.ID, drifted.ID); err == nil {
				t.Fatal("candidate activated with drifted " + name)
			}
		})
	}
	if _, err := ActivateCandidateCatalog(ctx, store, staging.Candidate.ID, grant.ID); err != nil {
		t.Fatal(err)
	}
}

func TestActiveCatalogExcludesStaleRegisteredAliases(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	first := catalogSearchManual(t, "first.tool", "Inspect the first surface.", "query")
	second := catalogSearchManual(t, "second.tool", "Inspect the second surface.", "query")
	activateTestCandidate(t, store, first, "first source")
	activateTestCandidate(t, store, second, "second source")
	if _, err := ResolveRegisteredManual(ctx, store, first.Name); err == nil {
		t.Fatal("stale registered alias remained active")
	}
	if _, err := ResolveRegisteredManual(ctx, store, second.Name); err != nil {
		t.Fatal(err)
	}
}

func activateTestCandidate(t *testing.T, store *overgodb.Store, manual Manual, sourceText string) {
	t.Helper()
	ctx := t.Context()
	source := testcheck.Must(openAPISourceContent(sourceText))(t)
	staging := testcheck.Must(StageManualCandidates(ctx, store, source, []Manual{manual}))(t)
	verification, _, err := PublishCandidateVerification(ctx, store, staging.Candidate.ID, NewOperatorExecutor())
	testcheck.NoError(t, err)
	if _, err := ActivateCandidateCatalog(ctx, store, staging.Candidate.ID, verification.ID); err != nil {
		t.Fatal(err)
	}
}

func openAPISourceContent(value string) (artifact.Content, error) {
	return (artifact.DocumentContract{
		Kind: artifact.KindFile, MediaType: artifact.JSONMediaType, Schema: "test/tool-source/v1",
	}).ContentBytes([]byte(value))
}
