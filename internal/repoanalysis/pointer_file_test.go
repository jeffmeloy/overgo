package repoanalysis

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// pointerText is the whole of AGENTS.md, which Codex reads, and CLAUDE.md imports it so Claude reads the same: where
// the written guidance is and the three commands of a step. A line belongs here only while code cannot hold it; once
// the driver runs every step, the commands and the last sentence go too.
const pointerText = "# overgo\n\nskill.md is the only written guidance; each of its rules names the check that holds it.\n\n" +
	"- Next step: `go run ./cmd/plan -next`\n- Check it: `go run ./cmd/plan -verify`\n" +
	"- Commit it: `go run ./cmd/loop -land <item>/<step> -message-file -`\n\n" +
	"When a command refuses, do what its message says and keep going. Don't stop to ask.\n"

// TestThePointerFileNamesSkillAndTheCommands holds the repository's instruction files to the pointer: AGENTS.md is
// pointerText and nothing else, and CLAUDE.md imports it and says nothing of its own, so no rule is written beside
// skill.md for one tool and not the other.
func TestThePointerFileNamesSkillAndTheCommands(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	read := func(name string) string { return string(testcheck.Must(os.ReadFile(filepath.Join(root, name)))(t)) }
	if agents := read("AGENTS.md"); agents != pointerText {
		t.Fatalf("AGENTS.md =\n%s\nwant the pointer alone:\n%s", agents, pointerText)
	}
	if claude := read("CLAUDE.md"); claude != "@AGENTS.md\n" {
		t.Fatalf("CLAUDE.md = %q, want the import of AGENTS.md alone", claude)
	}
}
