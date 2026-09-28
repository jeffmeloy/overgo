package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// formerCommands are the model-file commands each subcommand replaced.
var formerCommands = map[string]string{
	"hash": "gguf-hash", "inspect": "inspect-gguf", "merge": "gguf-merge",
	"quantize": "gguf-quantize", "split": "gguf-split",
}

// TestGGUFSubcommandsReplaceTheirCommands holds each subcommand, one subtest
// by its name, to the command it replaced: the command is gone and the tool
// is reached by name through gguf, refusing missing arguments with its own
// usage. An unknown name is refused with the subcommands listed.
func TestGGUFSubcommandsReplaceTheirCommands(t *testing.T) {
	t.Parallel()
	for name := range subcommands {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			former, known := formerCommands[name]
			if !known {
				t.Fatalf("gguf %s names no command it replaced", name)
			}
			if _, err := os.Stat(filepath.Join("..", former)); !os.IsNotExist(err) {
				t.Errorf("cmd/%s still exists beside gguf %s (%v)", former, name, err)
			}
			if err := runSubcommand([]string{name}); err == nil || !strings.Contains(err.Error(), "gguf "+name) {
				t.Errorf("gguf %s with no arguments = %v, want its usage", name, err)
			}
		})
	}
	t.Run("unknown", func(t *testing.T) {
		t.Parallel()
		if err := runSubcommand([]string{"absent"}); err == nil || !strings.Contains(err.Error(), "hash, inspect, merge") {
			t.Errorf("unknown subcommand = %v, want the subcommands listed", err)
		}
	})
}
