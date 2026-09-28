package main

import (
	"path/filepath"
	"strings"
	"testing"
)

// formerCommands are the store commands each subcommand replaced.
var formerCommands = map[string]string{
	"backup": "overgodb-backup", "check": "store-check", "import": "overgodb-import",
	"precheck": "store-precheck", "query": "overgodb-query", "rebuild": "overgodb-rebuild",
	"repair": "overgodb-repair",
}

// TestOvergoDBSubcommandsReplaceTheirCommands holds each store subcommand to
// the command it replaced: no Go source is left in the command's directory
// and the tool is reached by name through overgodb. An unknown name is
// refused with the subcommands listed.
func TestOvergoDBSubcommandsReplaceTheirCommands(t *testing.T) {
	t.Parallel()
	for name := range subcommands {
		former, known := formerCommands[name]
		if !known {
			t.Errorf("overgodb %s names no command it replaced", name)
			continue
		}
		if sources, _ := filepath.Glob(filepath.Join("..", former, "*.go")); len(sources) != 0 {
			t.Errorf("cmd/%s still holds %v beside overgodb %s", former, sources, name)
		}
	}
	if len(subcommands) != len(formerCommands) {
		t.Errorf("subcommands %d, replaced commands %d", len(subcommands), len(formerCommands))
	}
	if err := runSubcommand([]string{"absent"}); err == nil || !strings.Contains(err.Error(), "backup, check, import, precheck, query, rebuild, repair") {
		t.Errorf("unknown subcommand = %v, want the subcommands listed", err)
	}
}
