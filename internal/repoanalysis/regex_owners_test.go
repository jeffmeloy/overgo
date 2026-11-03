package repoanalysis

import (
	"maps"
	"slices"
	"strconv"
	"testing"

	"overgo/internal/gosyntax"
)

// The declared pattern-language owners: a file here matches text whose
// grammar is a pattern language by nature, not structured data a typed
// decoder owns. Removing the last regexp from a file removes it here.
var regexOwners = map[string]string{
	"cmd/device-lane/admission.go":             "compiles a go test -run selector, which is a regular expression",
	"cmd/device-lane/main.go":                  "compiles a go test -run selector, which is a regular expression",
	"cmd/webui-lane/main.go":                   "compiles a go test -run selector, which is a regular expression",
	"internal/cuda/testutil/require.go":        "quotes a test name into a go test -run selector",
	"internal/evaluation/ifeval_punkt.go":      "IFEval sentence rules are defined as patterns",
	"internal/evaluation/ifeval_structure.go":  "IFEval structure rules are defined as patterns",
	"internal/evaluation/ifeval_text.go":       "IFEval text rules are defined as patterns",
	"internal/evaluation/instruction_rules.go": "instruction-following rules are defined as patterns",
	"internal/guard/guard.go":                  "scans shell command text, a pattern language by nature",
	"internal/runrecord/verify_selector.go":    "compiles a go test -run selector, which is a regular expression",
	"internal/testevidence/evidence.go":        "quotes a test name into a go test -run selector",
}

// TestNoRegexParsesStructuredData holds every production regexp to a declared
// pattern-language owner: structured data -- identities, JSON, test output,
// git output -- is decoded by its owner, and a declaration whose file no
// longer uses regexp is stale.
func TestNoRegexParsesStructuredData(t *testing.T) {
	t.Parallel()
	snapshot, err := gosyntax.DiscoverGo("../..", "internal", "cmd")
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

// TestNoTextFormatIsReadByRegex: a C table, a PTX listing, a PDF object and a replacement string each have an exact form,
// so each is read by a parser, and none comes back as a declared owner.
func TestNoTextFormatIsReadByRegex(t *testing.T) {
	t.Parallel()
	for _, path := range []string{
		"cmd/gen-iq-tables/main.go", "cmd/kernel-manifest/main.go", "internal/evaluation/ifeval_tokenizer_profile.go",
		"internal/server/document_text.go", "internal/tokenizer/gennfc/main.go",
	} {
		if regexOwners[path] != "" {
			t.Errorf("%s reads a text format by regex", path)
		}
	}
}
