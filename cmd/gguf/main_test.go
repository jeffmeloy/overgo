package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// formerCommands are the model-file commands each subcommand replaced.
var formerCommands = map[string]string{
	"audio": "audio-inspect", "convert": "hf-gguf-convert", "convert-gemma4": "gemma4-gguf-convert", "hash": "gguf-hash",
	"inspect": "inspect-gguf", "merge": "gguf-merge", "quantize": "gguf-quantize",
	"safetensors": "inspect-safetensors", "split": "gguf-split",
}

// newSubcommands are the model-file tools that replaced no command: each was
// built as a subcommand, so there is no former command to be gone.
var newSubcommands = map[string]bool{"convert-lora": true, "merge-lora": true}

// TestGGUFSubcommandsReplaceTheirCommands holds each subcommand, one subtest
// by its name, to the command it replaced: the command is gone and the tool
// is reached by name through gguf, refusing missing arguments with its own
// usage. A subcommand that replaced none says so in newSubcommands. An unknown
// name is refused with every subcommand listed.
func TestGGUFSubcommandsReplaceTheirCommands(t *testing.T) {
	t.Parallel()
	for name := range subcommands {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			former, known := formerCommands[name]
			switch {
			case newSubcommands[name]:
				if known {
					t.Fatalf("gguf %s is listed as both new and a replacement of cmd/%s", name, former)
				}
			case !known:
				t.Fatalf("gguf %s names no command it replaced", name)
			default:
				if _, err := os.Stat(filepath.Join("..", former)); !os.IsNotExist(err) {
					t.Errorf("cmd/%s still exists beside gguf %s (%v)", former, name, err)
				}
			}
			if err := runSubcommand([]string{name}); err == nil || !strings.Contains(err.Error(), "gguf "+name) {
				t.Errorf("gguf %s with no arguments = %v, want its usage", name, err)
			}
		})
	}
	t.Run("unknown", func(t *testing.T) {
		t.Parallel()
		err := runSubcommand([]string{"absent"})
		if err == nil {
			t.Fatal("an unknown subcommand ran")
		}
		for name := range subcommands {
			if !strings.Contains(err.Error(), name) {
				t.Errorf("unknown subcommand = %v, want %s listed", err, name)
			}
		}
	})
}
