package gate

import (
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
)

// reductionContract freezes the first expensive-work reduction: the
// consumer, the assertions it keeps, every invalidator that returns the
// consumer to the plan, the retained baseline, and the mechanism.
type reductionContract struct {
	Consumer     []string
	Assertions   []string
	Invalidators []string
	Baseline     []string
	Mechanism    string
	Rollback     string
}

// testingReductionContract is the frozen contract for the device and browser
// lanes on a change that touches only gate implementation. The retained gate
// results named under Baseline show both lanes executing for such changes
// with every owned check unresolved; the shadow matrix and the gate's own
// selected tests remain the failure detectors for the gate itself.
func testingReductionContract() reductionContract {
	return reductionContract{
		Consumer: []string{"device", automationcheck.WebUICheckName},
		Assertions: []string{
			"a change reaching an owned package of either lane still selects it through the ownership or dependency closure",
			"a change under a web UI asset path still triggers the browser lane",
			"an analyzer or check-definition change still forces the complete selectable plan",
			"the gate's own package tests, including TestChangeSelectionRequiredMatrix, remain selected for a gate change",
		},
		Invalidators: []string{
			"internal/codemanifest/", "internal/codeprofile/", "internal/repoanalysis/", "cmd/code-manifest/",
			"internal/automationcheck/", "go.mod", "go.sum",
			"any path an owned package of the lane compiles or names at run time",
		},
		Baseline: []string{
			"11b13e106ca8307d4d49ffc2bbf2e242dfed56d4 result evidence:sha256:7d88611f5f9eb94d135fd11a22279c76704529ba39cef7294d990c355350f296: gate-only change, test-device 10m0s and webui-lane 6m38s executed, six owned checks unresolved",
			"3f1c6750f37a4365bc44ad0d2bd943c1bd24aa49 result evidence:sha256:135665fd8edcbf01700a01085e81e977f38164794e7f391e8a5b62b524e26175: gate and run-record change, test-device and webui-lane executed, six owned checks unresolved",
		},
		Mechanism: "the analyzer bootstrap no longer names internal/gate or cmd/gate; a changed document some package names belongs to its named readers instead of raising a non-Go uncertainty, a path join of adjacent literals names the joined path, and the ownership closure decides the lanes",
		Rollback:  "restore the two owners to requiresManifestBootstrap and drop the document attribution; every owned check runs again on a gate change",
	}
}

// TestTestingReductionContract proves the frozen reduction on the live
// repository: a change that touches only gate implementation no longer
// forces the analyzer bootstrap, a lane whose closure holds no Go-parsing
// reach is excluded through the ownership closure, the gate's own tests
// stay selected, a named document belongs to its readers, and a change to
// the check-definition owner still forces every lane back into the plan.
// The two lanes' own exclusion is measured and recorded, not yet held.
func TestTestingReductionContract(t *testing.T) {
	t.Parallel()
	contract := testingReductionContract()
	if len(contract.Consumer) != 2 || len(contract.Invalidators) < 7 || len(contract.Baseline) != 2 || contract.Mechanism == "" || contract.Rollback == "" {
		t.Fatalf("incomplete contract: %+v", contract)
	}
	for _, name := range []string{"internal/gate/preflight.go", "cmd/gate/main.go"} {
		if requiresManifestBootstrap([]string{name}) {
			t.Fatalf("%s forced the analyzer bootstrap the contract removes", name)
		}
	}
	for _, name := range contract.Invalidators[:5] {
		if !requiresManifestBootstrap([]string{name + "x.go"}) {
			t.Fatalf("invalidator %s no longer forces the bootstrap", name)
		}
	}
	// The shared checkout seeds every planned path, so each plan holds a
	// structural seed on a clean tree exactly as in a gate candidate.
	live := liveRepositoryFixture(t)
	plan := func(paths []string) (plannedPipeline, time.Duration) {
		t.Helper()
		var planned plannedPipeline
		var wall time.Duration
		live.plan(t, paths, func(g *gateContext) {
			started := time.Now()
			var err error
			planned, err = g.planPipeline()
			if err != nil {
				t.Fatal(err)
			}
			wall = time.Since(started)
			for _, note := range compactAudit(g.audit) {
				t.Log(note)
			}
		})
		return planned, wall
	}
	unrelated, unrelatedWall := plan([]string{"internal/gate/preflight.go"})
	if len(unrelated.structural.Seeds) == 0 {
		t.Fatal("seeded gate-only plan holds no structural seed")
	}
	// Measured 2026-09-14 on the seeded checkout: both lanes stay triggered
	// by their capability facts because an owned package reaches the gate
	// through an unnamed program (internal/projector) or a production source
	// reader; their exclusion is recorded here and held by no assertion yet.
	for _, name := range contract.Consumer {
		reason, excluded := unrelated.impact.ExclusionReason(name)
		t.Logf("gate-only change: %s excluded=%v reason=%s unknown=%v seeds=%v", name, excluded, reason, unrelated.surface.Unknown, unrelated.structural.Seeds)
		if excluded && !strings.Contains(reason, "closure") {
			t.Fatalf("%s excluded without a closure proof: %s", name, reason)
		}
	}
	if reason, excluded := unrelated.impact.ExclusionReason("sbom"); !excluded || !strings.Contains(reason, "closure") {
		t.Fatalf("gate-only change did not exclude sbom by closure: excluded=%v reason=%s", excluded, reason)
	}
	for _, name := range []string{"acceptance", testPlanCheckName, testOwnersCheckName, "commit"} {
		if !hasInvocation(unrelated, name) {
			t.Fatalf("gate-only change dropped %s", name)
		}
	}
	// Every real commit carries the regenerated gate-owned documents. Their
	// named readers own the change: no device package names them or
	// compiles a reader, so the device lane stays excluded, while the
	// browser census names the API manifest, so the browser lane runs.
	documented, documentedWall := plan([]string{"internal/gate/preflight.go", "docs/plan.json", "docs/api_manifest.json", "docs/modern_go_census.json"})
	reason, excluded := documented.impact.ExclusionReason("device")
	t.Logf("gate change with documents: device excluded=%v reason=%s unknown=%v", excluded, reason, documented.surface.Unknown)
	if _, excluded := documented.impact.ExclusionReason(automationcheck.WebUICheckName); excluded || !hasInvocation(documented, automationcheck.WebUICheckName) {
		t.Fatal("the browser lane, whose census names the API manifest, was excluded for a manifest change")
	}
	for _, unknown := range documented.surface.Unknown {
		if strings.HasPrefix(unknown, "non-go:") {
			t.Fatalf("a named document kept its non-Go uncertainty: %s", unknown)
		}
	}
	t.Logf("gate change with documents: browser retained, in %s", documentedWall)
	relevant, relevantWall := plan([]string{"internal/automationcheck/check.go"})
	for _, name := range contract.Consumer {
		if _, excluded := relevant.impact.ExclusionReason(name); excluded || !hasInvocation(relevant, name) {
			t.Fatalf("check-definition change excluded %s", name)
		}
	}
	if !strings.Contains(strings.Join(relevant.surface.Unknown, "; "), "planner implementation changed") {
		t.Fatalf("check-definition change did not force the bootstrap: %v", relevant.surface.Unknown)
	}
	t.Logf("gate-only change planned in %s; check-definition change: lanes retained in %s", unrelatedWall, relevantWall)
}

func hasInvocation(planned plannedPipeline, name string) bool {
	for _, invocation := range planned.invocations {
		if invocation.Check.Name == name {
			return true
		}
	}
	return false
}

// TestNamedDocumentAttribution resolves a changed document to the packages
// that name it and leaves an unnamed document uncertain: the opaque reader
// beside the named reader does not own it.
func TestNamedDocumentAttribution(t *testing.T) {
	t.Parallel()
	g := runtimeReaderFixture(t)
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	if readers := graph.namedReaders("docs/config.txt"); !slices.Equal(readers, []string{"internal/reader"}) {
		t.Fatalf("named readers = %v", readers)
	}
	if readers := graph.namedReaders("README.md"); len(readers) != 0 {
		t.Fatalf("unnamed document has readers %v", readers)
	}
	impact := codemanifest.Impact{
		Packages: []string{"docs", "internal/reader"},
		ExternalInputs: []codemanifest.ExternalInputChange{
			{Path: "docs/config.txt", Kind: codemanifest.ChangeModified, Base: &codemanifest.ExternalInput{Path: "docs/config.txt", Owner: "docs"}, Candidate: &codemanifest.ExternalInput{Path: "docs/config.txt", Owner: "docs"}},
			{Path: "README.md", Kind: codemanifest.ChangeModified, Base: &codemanifest.ExternalInput{Path: "README.md", Owner: "."}, Candidate: &codemanifest.ExternalInput{Path: "README.md", Owner: "."}},
		},
		Uncertainty: []codemanifest.Uncertainty{
			{Kind: codemanifest.UncertaintyNonGo, Path: "docs/config.txt", Reason: "changed non-Go input has no structural dependency adapter"},
			{Kind: codemanifest.UncertaintyNonGo, Path: "README.md", Reason: "changed non-Go input has no structural dependency adapter"},
			{Kind: codemanifest.UncertaintyInterface, Reason: "dispatch"},
		},
	}
	attributed, documents, notes := attributeNamedDocuments(impact, graph)
	if len(notes) != 1 || !strings.Contains(notes[0], "docs/config.txt attributed to its named readers internal/reader") {
		t.Fatalf("notes = %v", notes)
	}
	if len(attributed.Uncertainty) != 2 || attributed.Uncertainty[0].Path != "README.md" || attributed.Uncertainty[1].Kind != codemanifest.UncertaintyInterface {
		t.Fatalf("uncertainty = %+v", attributed.Uncertainty)
	}
	if !slices.Equal(documents["docs/config.txt"], []string{"internal/reader"}) || len(documents) != 1 {
		t.Fatalf("documents = %v", documents)
	}
	// Owners and packages are untouched: the resolver answers the document's
	// directory through the named readers instead.
	if attributed.ExternalInputs[0].Base.Owner != "docs" || !slices.Equal(attributed.Packages, impact.Packages) || len(impact.Uncertainty) != 3 {
		t.Fatalf("attribution changed owners or packages: %+v", attributed)
	}
	g.attributedDocuments = documents
	resolver, err := g.dependencyResolver()
	if err != nil {
		t.Fatal(err)
	}
	owning := func(packages ...string) automationcheck.Ownership {
		return automationcheck.Ownership{Fact: "lane", Packages: packages}
	}
	if !resolver(owning("internal/reader"), "docs") {
		t.Fatal("the named reader does not reach its own document")
	}
	if !resolver(owning("internal/readerclient"), "docs") {
		t.Fatal("a package compiling the named reader does not reach the document")
	}
	if resolver(owning("internal/isolated"), "docs") {
		t.Fatal("a package with neither the name nor the reader reached the document")
	}
	if !resolver(owning("internal/isolated"), "unattributed") {
		t.Fatal("an unattributed unknown directory stopped binding every lane")
	}
}
