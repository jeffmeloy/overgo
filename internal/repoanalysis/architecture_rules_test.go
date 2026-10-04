// Package repoanalysis_test holds the live-tree checks of the repository's
// architecture rules.
package repoanalysis_test

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/repoanalysis"
)

// TestEntryAuthoritiesHoldOverTheTree holds the whole tree to the entry
// authorities: storage, tools, promotion and the Go-only runtime each have
// one owner, and a script named as a runtime entry is refused.
func TestEntryAuthoritiesHoldOverTheTree(t *testing.T) {
	t.Parallel()
	var rules []repoanalysis.EntryAuthorityRule
	for _, domain := range []repoanalysis.EntryAuthorityDomain{
		repoanalysis.EntryAuthorityStorage, repoanalysis.EntryAuthorityTool,
		repoanalysis.EntryAuthorityPromotion, repoanalysis.EntryAuthorityGoOnly,
	} {
		rule, found := repoanalysis.EntryAuthorityRuleFor(domain)
		if !found {
			t.Fatalf("the entry authority table lost the %s domain", domain)
		}
		rules = append(rules, rule)
	}
	if _, err := repoanalysis.ValidateEntryAuthorities(architectureRSISnapshot(t), rules); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte("package rogue\n\nconst deployHook = \"deploy.ps1\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := gosyntax.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	goOnly, _ := repoanalysis.EntryAuthorityRuleFor(repoanalysis.EntryAuthorityGoOnly)
	if _, err := repoanalysis.ValidateEntryAuthorities(violation, []repoanalysis.EntryAuthorityRule{goOnly}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("runtime script literal was not refused: %v", err)
	}
	architectureRSIFinding(t, "internal/loop/rogue_markdown.go", "package loop\nimport \"os\"\nfunc loadBehavior() { _, _ = os.ReadFile(\"skill.md\") }\n", "go-only", "non-go-behavior")
}

func TestProductionAuthorityBoundaries(t *testing.T) {
	t.Parallel()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := gosyntax.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	clean, err := repoanalysis.AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if err := clean.Error(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		family  string
		overlay map[string][]byte
	}{
		{name: "storage", family: "storage", overlay: map[string][]byte{
			"internal/server/rogue_producer.go": []byte("package server\nimport \"overgo/internal/overgodb\"\nfunc rogueProducer() { _ = overgodb.NewProducer() }\n"),
		}},
		{name: "aliased process", family: "process", overlay: map[string][]byte{
			"internal/server/rogue_process.go": []byte("package server\nimport runner \"os/exec\"\nfunc rogueProcess() { _ = runner.Command(\"probe\") }\n"),
		}},
		{name: "capability", family: "capability", overlay: map[string][]byte{
			"internal/server/rogue_capability.go": []byte("package server\nimport capability \"overgo/internal/capabilityruntime\"\nvar roguePlacement = capability.ExactCapabilityPlacement{Task: \"rogue\"}\n"),
		}},
		{name: "active catalog bound through the alias owner", family: "tool", overlay: map[string][]byte{
			"internal/server/rogue_catalog.go": []byte("package server\nimport (\"overgo/internal/agenttool\"; \"overgo/internal/artifact\")\nfunc rogueCatalog(target, held artifact.ID) artifact.AliasBinding { return artifact.AliasMove(agenttool.ActiveCatalogAlias, target, held) }\n"),
		}},
		{name: "trigger", family: "trigger", overlay: map[string][]byte{
			"internal/server/rogue_trigger.go": []byte("package server\nimport records \"overgo/internal/runrecord\"\nvar rogueTrigger = records.CausalContext{Trigger: records.TriggerWebhook}\n"),
		}},
		{name: "promotion", family: "promotion", overlay: map[string][]byte{
			"internal/server/rogue_promotion.go": []byte("package server\nimport comp \"overgo/internal/composition\"\nfunc roguePromotion() { _, _, _, _ = comp.ActiveCompositeGeneration(nil, nil, artifact.ID{}, artifact.ID{}, recipe.Task{}) }\n"),
		}},
		{name: "stale allowance", family: "process", overlay: map[string][]byte{
			"cmd/sbom/main.go": nil,
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate, err := snapshot.Overlay(test.overlay)
			if err != nil {
				t.Fatal(err)
			}
			report, err := repoanalysis.AuditProductionAuthorityBoundaries(candidate)
			if err != nil {
				t.Fatal(err)
			}
			if report.Error() == nil {
				t.Fatalf("%s bypass was admitted", test.family)
			}
			if !slices.ContainsFunc(report.Findings, func(finding repoanalysis.ProductionAuthorityFinding) bool {
				return finding.Family == test.family
			}) {
				t.Fatalf("%s bypass produced findings for the wrong family: %+v", test.family, report.Findings)
			}
		})
	}
}

func TestArchitectureRatchetClassifiesCapabilityInvocationBoundaries(t *testing.T) {
	t.Parallel()
	architectureRSIFinding(t, "internal/server/rogue_invocation.go", "package server\nimport tools \"overgo/internal/agenttool\"\nfunc invoke(e *tools.Executor) { _, _, _ = e.InvokeWithEffect(nil, tools.Manual{}, nil) }\n", "invocation-boundary", "unclassified")
}

func TestInternalOrchestrationDoesNotInvokeUTCPTransport(t *testing.T) {
	t.Parallel()
	architectureRSIFinding(t, "internal/loop/rogue_transport.go", "package loop\nimport tools \"overgo/internal/agenttool\"\nfunc routeInternally(e *tools.Executor) { _, _, _ = e.InvokeWithEffect(nil, tools.Manual{}, nil) }\n", "direct-go", "transport-bypass")
}

func TestRSIRuntimeIsGoOnly(t *testing.T) {
	t.Parallel()
	snapshot := architectureRSISnapshot(t)
	tests := []struct {
		name string
		path string
		body string
		kind string
	}{
		{
			name: "runtime loader", path: "internal/loop/rogue_plugin.go", kind: "runtime-loader",
			body: "package loop\nimport \"plugin\"\nfunc load() { _, _ = plugin.Open(\"mechanism.so\") }\n",
		},
		{
			name: "interpreter", path: "internal/loop/rogue_python.go", kind: "interpreted-runtime",
			body: "package loop\nimport \"os/exec\"\nfunc run() { _ = exec.Command(\"python\", \"mechanism.py\").Run() }\n",
		},
		{
			name: "markdown behavior", path: "internal/loop/rogue_skill.go", kind: "non-go-behavior",
			body: "package loop\nimport \"os\"\nfunc load() { _, _ = os.ReadFile(\"SKILL.md\") }\n",
		},
		{
			name: "filesystem catalog", path: "internal/agenttool/rogue_catalog.go", kind: "filesystem-capability-discovery",
			body: "package agenttool\nimport \"os\"\nfunc discover() { _, _ = os.ReadDir(\"capabilities\") }\n",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := architectureRSIOverlay(t, snapshot, map[string][]byte{test.path: []byte(test.body)})
			report := architectureRSIAudit(t, candidate)
			requireArchitectureFinding(t, report, "go-only", test.kind)
		})
	}
}

func architectureRSIFinding(t *testing.T, path, body, family, kind string) {
	t.Helper()
	snapshot := architectureRSISnapshot(t)
	candidate := architectureRSIOverlay(t, snapshot, map[string][]byte{path: []byte(body)})
	report := architectureRSIAudit(t, candidate)
	requireArchitectureFinding(t, report, family, kind)
}

func architectureRSISnapshot(t *testing.T) gosyntax.Snapshot {
	t.Helper()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := gosyntax.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func architectureRSIOverlay(t *testing.T, snapshot gosyntax.Snapshot, overlay map[string][]byte) gosyntax.Snapshot {
	t.Helper()
	candidate, err := snapshot.Overlay(overlay)
	if err != nil {
		t.Fatal(err)
	}
	return candidate
}

func architectureRSIAudit(t *testing.T, snapshot gosyntax.Snapshot) repoanalysis.ProductionAuthorityReport {
	t.Helper()
	report, err := repoanalysis.AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	return report
}

func requireArchitectureFinding(t *testing.T, report repoanalysis.ProductionAuthorityReport, family, kind string) {
	t.Helper()
	for _, finding := range report.Findings {
		if finding.Family == family && finding.Kind == kind {
			return
		}
	}
	t.Fatalf("missing %s/%s finding: %+v", family, kind, report.Findings)
}
