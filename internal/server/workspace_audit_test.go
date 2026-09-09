package server

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// workspaceFieldOwners maps every promoted workspace field to the
// dependency struct that owns it. The shared core fields are reachable
// from every file and are not listed.
var workspaceFieldOwners = map[string]string{
	"generator": "serving", "sessions": "serving", "defaultSampling": "serving",
	"defaultOutputTokens": "serving", "slotBusy": "serving", "slotTasks": "serving",
	"slotStats": "serving", "nextTask": "serving", "nextID": "serving",
	"generatedTokens": "serving", "mediaFetcher": "serving", "responseFiles": "serving",
	"thinkingSigner": "serving",
	"operations":     "operator", "tools": "operator", "issuedCalls": "operator",
	"agentCoordinator": "agent", "agentSessions": "agent",
	"downloads":   "hub",
	"catalogMemo": "workbench", "browseRepository": "workbench",
}

// workspaceFileAllowances is the reviewed coupling baseline: the exact
// workspaces each production file may reach through the handler. A file
// absent from the table may touch only the shared core. Widening a row is
// a reviewed edit; the audit refuses silent new couplings, so no workspace
// drifts back into reaching the entire handler.
var workspaceFileAllowances = map[string][]string{
	"agent_control.go":               {"agent", "operator", "serving"},
	"agent_workspace.go":             {"agent"},
	"analyze_attention.go":           {"serving"},
	"analyze_model.go":               {"serving"},
	"analyze_states.go":              {"serving"},
	"analyze_tensors.go":             {"serving"},
	"analyze_vocab.go":               {"serving"},
	"anthropic_tools.go":             {"serving"},
	"artifact_gallery.go":            {"workbench"},
	"artifact_lineage.go":            {"serving"},
	"prompt_enhance.go":              {"serving", "workbench"},
	"automation_routes.go":           {"operator", "serving"},
	"automation_webhook.go":          {"operator", "serving"},
	"capability_bundles.go":          {"operator"},
	"evaluation_workspace.go":        {"operator"},
	"generation_workspace.go":        {"operator", "serving"},
	"hub_workspace.go":               {"hub", "workbench"},
	"library_workspace.go":           {"operator"},
	"native_media_protocols.go":      {"operator", "serving"},
	"native_audio_transcriptions.go": {"serving"},
	"streaming_transcriptions.go":    {"serving"},
	"operation_evidence.go":          {"operator"},
	"operations.go":                  {"operator"},
	"operator_decisions.go":          {"operator"},
	"peer_routes.go":                 {"operator", "serving"},
	"protocol_anthropic.go":          {"serving"},
	"protocol_chat.go":               {"operator", "serving"},
	"protocol_common.go":             {"serving"},
	"protocol_responses.go":          {"serving"},
	"recipe_inspector.go":            {"serving"},
	"response_interaction.go":        {"operator", "serving"},
	"responses_tools.go":             {"operator", "serving"},
	"runtime_activity.go":            {"operator", "serving"},
	"server.go":                      {"hub", "operator", "serving", "workbench"},
	"server_admin.go":                {"serving"},
	"server_completion.go":           {"serving"},
	"server_embeddings.go":           {"serving"},
	"server_native_generation.go":    {"serving"},
	"server_native_options.go":       {"serving"},
	"server_native_prompt.go":        {"serving"},
	"server_native_types.go":         {"serving"},
	"serving_observation.go":         {"serving"},
	"tool_execution.go":              {"operator"},
	"workspace_capabilities.go":      {"serving"},
	"workspace_manifest.go":          {"agent", "serving"},
}

func TestWorkspaceDependencyBoundaries(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	violations := []string{}
	observed := map[string]map[string]bool{}
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(token.NewFileSet(), name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, declaration := range parsed.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || function.Recv == nil || len(function.Recv.List) != 1 {
				continue
			}
			receiver := receiverIdent(function)
			if receiver == "" || !handlerReceiver(function) {
				continue
			}
			ast.Inspect(function.Body, func(node ast.Node) bool {
				selector, ok := node.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				base, ok := selector.X.(*ast.Ident)
				if !ok || base.Name != receiver {
					return true
				}
				workspace, owned := workspaceFieldOwners[selector.Sel.Name]
				if !owned {
					return true
				}
				if observed[name] == nil {
					observed[name] = map[string]bool{}
				}
				observed[name][workspace] = true
				return true
			})
		}
	}
	for file, workspaces := range observed {
		allowed := map[string]bool{}
		for _, workspace := range workspaceFileAllowances[filepath.ToSlash(file)] {
			allowed[workspace] = true
		}
		for workspace := range workspaces {
			if !allowed[workspace] {
				violations = append(violations, file+" -> "+workspace)
			}
		}
	}
	slices.Sort(violations)
	for _, violation := range violations {
		t.Errorf("undeclared workspace coupling: %s", violation)
	}
}

func receiverIdent(function *ast.FuncDecl) string {
	names := function.Recv.List[0].Names
	if len(names) != 1 {
		return ""
	}
	return names[0].Name
}

func handlerReceiver(function *ast.FuncDecl) bool {
	expression := function.Recv.List[0].Type
	if star, ok := expression.(*ast.StarExpr); ok {
		expression = star.X
	}
	ident, ok := expression.(*ast.Ident)
	return ok && ident.Name == "Handler"
}
