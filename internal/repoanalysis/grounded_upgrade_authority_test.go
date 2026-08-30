package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestGroundedCapabilityUpgradeUsesExistingAuthorities(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	report, err := AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if err := report.Error(); err != nil {
		t.Fatalf("grounded upgrade crossed a production authority: %v; findings=%+v", err, report.Findings)
	}

	for _, relative := range []string{
		"internal/memory", "internal/knowledge", "internal/vectorstore", "internal/graphstore",
		"internal/capabilityrouter", "cmd/memory", "cmd/vectorstore", "cmd/graphstore",
	} {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(relative))); !os.IsNotExist(err) {
			t.Fatalf("grounded upgrade created parallel subsystem %s", relative)
		}
	}

	owners := map[string]string{
		"CapabilityEpisodeProjection": "internal/dataset",
		"InteractionArc":              "internal/dataset",
		"DatasetTransform":            "internal/dataset",
		"RetrievalReceipt":            "internal/runrecord",
		"RoutingDecision":             "internal/runrecord",
		"CapabilityProbeResult":       "internal/evaluation",
		"TrajectoryHealthDecision":    "internal/evaluation",
		"EpisodeKnowledgeProposal":    "internal/evaluation",
	}
	seen := make(map[string]string, len(owners))
	for _, source := range snapshot.Files {
		if source.Test {
			continue
		}
		file, parseErr := source.Syntax()
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		for _, declaration := range file.Decls {
			generic, ok := declaration.(*ast.GenDecl)
			if !ok || generic.Tok != token.TYPE {
				continue
			}
			for _, raw := range generic.Specs {
				typeSpec := raw.(*ast.TypeSpec)
				expected, protected := owners[typeSpec.Name.Name]
				if !protected {
					continue
				}
				if prior, duplicate := seen[typeSpec.Name.Name]; duplicate {
					t.Fatalf("authority %s duplicated in %s and %s", typeSpec.Name.Name, prior, source.Path)
				}
				actual := filepath.ToSlash(filepath.Dir(source.Path))
				if actual != expected {
					t.Fatalf("authority %s owned by %s, want %s", typeSpec.Name.Name, actual, expected)
				}
				seen[typeSpec.Name.Name] = source.Path
			}
		}
	}
	for name := range owners {
		if seen[name] == "" {
			t.Fatalf("grounded upgrade authority %s is absent", name)
		}
	}

	for _, relative := range []string{
		"internal/dataset/capability_episode.go", "internal/dataset/interaction_arc.go",
		"internal/dataset/agent_retrieval.go", "internal/dataset/transform.go",
		"internal/runrecord/dataset_projection_adapter.go", "internal/runrecord/retrieval_receipt.go",
		"internal/runrecord/routing_decision.go", "internal/evaluation/production_probe.go",
		"internal/evaluation/evidence_route.go", "internal/evaluation/trajectory_supervisor.go",
		"internal/evaluation/retrieval_quality.go", "internal/evaluation/knowledge_promotion.go",
	} {
		path := filepath.Join(root, filepath.FromSlash(relative))
		file, parseErr := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		for _, imported := range file.Imports {
			value, unquoteErr := strconv.Unquote(imported.Path.Value)
			if unquoteErr != nil {
				t.Fatal(unquoteErr)
			}
			if strings.Contains(value, ".") && !strings.HasPrefix(value, "overgo/internal/") {
				t.Fatalf("grounded upgrade file %s imports external runtime %s", relative, value)
			}
		}
	}

	serverSource, err := os.ReadFile(filepath.Join(root, "internal", "server", "agent_control.go"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(serverSource), "runrecord.SearchAndPublishConsumedRetrieval") {
		t.Fatal("production agent retrieval bypasses the search-and-receipt owner")
	}
	receiptSource, err := os.ReadFile(filepath.Join(root, "internal", "runrecord", "retrieval_receipt.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, rawExport := range []string{"func NewRetrievalReceipt", "func PublishConsumedRetrievalReceipt"} {
		if strings.Contains(string(receiptSource), rawExport) {
			t.Fatalf("caller-authored retrieval evidence remains exported: %s", rawExport)
		}
	}
}
