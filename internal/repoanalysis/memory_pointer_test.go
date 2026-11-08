package repoanalysis

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheProjectKeepsNoSessionMemory holds the project's session memory to its one pointer: every session is told to
// write memory by itself, and memory cut back grew rules again beside skill.md. Memory of the pointer alone reports
// nothing; each line the index gains, and each file beside it, is named; and the session-start hook says it.
func TestTheProjectKeepsNoSessionMemory(t *testing.T) {
	t.Parallel()
	pointer := "- [Guidance](skill.md) - the written rules, each naming its check\n"
	if beyond := MemoryBeyondPointer(map[string]string{"memory/MEMORY.md": pointer}); len(beyond) != 0 {
		t.Fatalf("memory of the pointer alone reports %q", beyond)
	}
	grown := map[string]string{"memory/MEMORY.md": pointer + "- [Lessons](lessons.md) - lessons that generalize\n", "memory/lessons.md": "a rule\n"}
	if beyond := MemoryBeyondPointer(grown); !slices.Equal(beyond, []string{"memory/MEMORY.md: - [Lessons](lessons.md) - lessons that generalize", "memory/lessons.md"}) {
		t.Fatalf("memory grown beyond its pointer reports %q, want the index's second line and the file beside it", beyond)
	}
	hook := string(testcheck.Must(os.ReadFile(filepath.Join("..", "..", "cmd", "loophook", "main.go")))(t))
	if !strings.Contains(hook, "repoanalysis.MemoryBeyondPointer(") {
		t.Fatal("the session-start hook does not report memory beyond its pointer")
	}
}
