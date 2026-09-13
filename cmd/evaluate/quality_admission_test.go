package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/longform"
	"overgo/internal/testutil"
)

func TestDerivedQualityRunsWithoutPerformanceQualification(t *testing.T) {
	root := testutil.RepoRoot(t)
	t.Chdir(root)
	repository, declaration := declareHostedFixture(t, "OVERGO_QUALITY_ADMISSION_TEST_KEY", []string{"A"})
	session, err := openEvaluationSession(t.Context(), manifest{Repository: repository, CodeCommit: hostedTestCommit}, modelRequest{Path: declaration.Location, Suites: []string{"derived"}})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	hosted := session.(*hostedSession)
	// Reuse the local suite orchestration with a CPU fixture runtime.
	native := nativeSession{store: hosted.store, campaign: hosted.campaign, model: declaration.Model}
	fixture, err := os.ReadFile(filepath.Join(root, "internal", "dataset", "testdata", "mmlu-dev.arrow"))
	if err != nil {
		t.Fatal(err)
	}
	cache := t.TempDir()
	target := filepath.Join(cache, "cais___mmlu", "abstract_algebra", "0.0.0", "aa11", "mmlu-test.arrow")
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, fixture, clioptions.OutputFileMode); err != nil {
		t.Fatal(err)
	}
	if _, _, err := evaluation.CatalogHFCacheBenchmarks(t.Context(), hosted.store, cache); err != nil {
		t.Fatal(err)
	}
	if err := longform.Admit(t.Context(), hosted.store, "", "unmeasured"); !errors.Is(err, longform.ErrRefused) {
		t.Fatalf("fixture unexpectedly performance-qualified: %v", err)
	}
	suites, dropped, err := evaluation.DeriveStoreSuiteFamily(t.Context(), hosted.store, hosted.campaign.Authorities(), "mmlu")
	if err != nil || len(suites) != 1 || len(dropped) != 0 {
		t.Fatalf("quality inputs: suites=%d dropped=%v err=%v", len(suites), dropped, err)
	}
	if err := native.EvaluateDerived(t.Context(), "mmlu"); err != nil {
		t.Fatal(err)
	}
	planID := suites[0].Plan().Identity()
	reportID, found, err := hosted.store.ResolveAlias(t.Context(), "evaluation/campaigns/"+planID.String())
	if err != nil || !found {
		t.Fatalf("quality report absent: found=%t err=%v", found, err)
	}
	content, err := artifact.RequireTypedContent(t.Context(), hosted.store, reportID)
	if err != nil {
		t.Fatal(err)
	}
	var report evaluation.MultipleChoiceReport
	if err := json.Unmarshal(content.Data, &report); err != nil {
		t.Fatal(err)
	}
	if report.Plan != planID || len(report.Observations) == 0 || uint64(len(report.Observations)) != suites[0].Descriptor().Cases {
		t.Fatalf("incomplete quality report: %+v", report)
	}
	if err := longform.Admit(t.Context(), hosted.store, "", "unmeasured"); !errors.Is(err, longform.ErrRefused) {
		t.Fatalf("quality run granted performance credit: %v", err)
	}
	if err := native.EvaluateDerived(t.Context(), "absent-family"); err == nil {
		t.Fatal("missing quality suite was accepted")
	}
}
