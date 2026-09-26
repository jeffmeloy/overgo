package repoanalysis

import (
	"maps"
	"slices"
	"strconv"
	"testing"
)

// The declared pattern-language owners: a file here matches text whose
// grammar is a pattern language by nature, not structured data a typed
// decoder owns. Removing the last regexp from a file removes it here.
var regexOwners = map[string]string{
	"cmd/build-kernels/main.go":                       "rewrites a generated digest constant in Go source text; follow-up: go/ast edit",
	"cmd/compatibility/webui_report.go":               "scans the shell's JavaScript and HTML text; the GUI lane replaces it with a browser census",
	"cmd/coverage-lanes/main.go":                      "reads go test's coverage summary line; follow-up: -coverprofile",
	"cmd/device-lane/admission.go":                    "compiles a go test -run selector, which is a regular expression",
	"cmd/device-lane/main.go":                         "compiles a go test -run selector, which is a regular expression",
	"cmd/gen-iq-tables/main.go":                       "generator reading C table source",
	"cmd/kernel-manifest/main.go":                     "reads PTX assembly entries, a text grammar with no Go decoder",
	"cmd/release/main.go":                             "validates a semantic-version grammar",
	"cmd/validate/media_receipt.go":                   "names acceptance test families as go test -run selectors",
	"cmd/webui-lane/main.go":                          "compiles a go test -run selector, which is a regular expression",
	"internal/agenttool/manual.go":                    "validates a manual-name grammar",
	"internal/closurescan/scan.go":                    "splits identifiers into words",
	"internal/cuda/testutil/require.go":               "quotes a test name into a go test -run selector",
	"internal/evaluation/ifeval_punkt.go":             "IFEval sentence rules are defined as patterns",
	"internal/evaluation/ifeval_structure.go":         "IFEval structure rules are defined as patterns",
	"internal/evaluation/ifeval_text.go":              "IFEval text rules are defined as patterns",
	"internal/evaluation/ifeval_tokenizer_profile.go": "IFEval tokenizer profile is defined as patterns",
	"internal/evaluation/instruction_rules.go":        "instruction-following rules are defined as patterns",
	"internal/gate/runtime_inputs.go":                 "classifies function names by verb",
	"internal/gemma4convert/convert.go":               "reads the Hugging Face tensor-name grammar",
	"internal/gemma4convert/tower.go":                 "reads the Hugging Face tensor-name grammar",
	"internal/guard/guard.go":                         "scans shell command text, a pattern language by nature",
	"internal/hfconvert/convert.go":                   "reads the Hugging Face tensor-name grammar",
	"internal/hfgguf/qwen35_projector.go":             "reads the Hugging Face tensor-name grammar",
	"internal/inference/chat_tool_grammar.go":         "validates a tool-name grammar",
	"internal/plan/plan.go":                           "polices doctrine prose for metric literals",
	"internal/repoanalysis/docs_inventory.go":         "reads markdown links and command references in prose",
	"internal/repoanalysis/family_branch_census.go":   "classifies model-family stems",
	"internal/runrecord/verify_selector.go":           "compiles a go test -run selector, which is a regular expression",
	"internal/sampling/json_schema.go":                "JSON Schema pattern and format keywords are regular expressions",
	"internal/server/document_text.go":                "locates PDF stream dictionaries in raw PDF bytes",
	"internal/server/resource_policy.go":              "validates a file-id grammar",
	"internal/testevidence/evidence.go":               "quotes a test name into a go test -run selector",
	"internal/tokenizer/gennfc/main.go":               "generator reading Unicode data tables",
	"internal/webuilane/census.go":                    "scans the shell's JavaScript text; the GUI lane replaces it with a browser census",
	"internal/webuilane/design.go":                    "reads the shell's CSS custom properties; the GUI lane replaces it with CSSOM",
}

// TestNoRegexParsesStructuredData holds every production regexp to a declared
// pattern-language owner: structured data -- identities, JSON, test output,
// git output -- is decoded by its owner, and a declaration whose file no
// longer uses regexp is stale.
func TestNoRegexParsesStructuredData(t *testing.T) {
	t.Parallel()
	snapshot, err := DiscoverGo("../..", "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	importing := map[string]bool{}
	for _, file := range snapshot.Files {
		if file.Test {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			t.Fatal(err)
		}
		for _, spec := range syntax.Imports {
			if path, _ := strconv.Unquote(spec.Path.Value); path == "regexp" {
				importing[file.Path] = true
			}
		}
	}
	for _, path := range slices.Sorted(maps.Keys(importing)) {
		if regexOwners[path] == "" {
			t.Errorf("%s uses regexp without a declared pattern-language owner; decode its structured input instead", path)
		}
	}
	for _, path := range slices.Sorted(maps.Keys(regexOwners)) {
		if !importing[path] {
			t.Errorf("%s is declared a pattern-language owner but no longer uses regexp; drop the declaration", path)
		}
	}
}
