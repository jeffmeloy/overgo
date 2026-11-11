package repoanalysis

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// pointerText is the whole of AGENTS.md, which Codex reads, and CLAUDE.md imports it so Claude reads the same: where
// the written guidance is and the three commands of a step. A line belongs here only while code cannot hold it; once
// the driver runs every step, the commands and the last sentence go too.
const pointerText = "# overgo\n\nskill.md is the only written guidance; each of its rules names the check that holds it.\n\n" +
	"- Next step: `go run ./cmd/plan -next`\n- Check it: `go run ./cmd/plan -verify`\n" +
	"- Commit it: `go run ./cmd/loop -land <item>/<step>`\n\n" +
	"When a command refuses, do what its message says and keep going. Don't stop to ask.\n"

// firstDifference says where got first parts from want: the line, each side quoted, so a space or a missing final
// newline, which two printed files do not show, does. A file that only prints alike told the local worker nothing of
// what was left to change (loop-proof/local-demonstrated, ask-nothing/pointer, runs 4 to 6).
func firstDifference(got, want string) string {
	gotLines, wantLines := strings.SplitAfter(got, "\n"), strings.SplitAfter(want, "\n")
	for index := range max(len(gotLines), len(wantLines)) {
		var have, need string
		if index < len(gotLines) {
			have = gotLines[index]
		}
		if index < len(wantLines) {
			need = wantLines[index]
		}
		if have != need {
			return fmt.Sprintf("line %d is %q, want %q", index+1, have, need)
		}
	}
	return "the two are the same"
}

// TestThePointerFileNamesSkillAndTheCommands holds the repository's instruction files to the pointer: AGENTS.md is
// pointerText and nothing else, and CLAUDE.md imports it and says nothing of its own, so no rule is written beside
// skill.md for one tool and not the other.
func TestThePointerFileNamesSkillAndTheCommands(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	read := func(name string) string { return string(testcheck.Must(os.ReadFile(filepath.Join(root, name)))(t)) }
	if agents := read("AGENTS.md"); agents != pointerText {
		t.Fatalf("AGENTS.md is not the pointer alone: %s\nAGENTS.md =\n%s\nwant:\n%s", firstDifference(agents, pointerText), agents, pointerText)
	}
	if claude := read("CLAUDE.md"); claude != "@AGENTS.md\n" {
		t.Fatalf("CLAUDE.md = %q, want the import of AGENTS.md alone", claude)
	}
}
