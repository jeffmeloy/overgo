package main

import (
	"bytes"
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// formerParityCommands are the commands each parity replaced.
var formerParityCommands = map[string]string{
	"audio-oracle": "audio-oracle", "sensenova": "sensenovaparity", "vqa": "vqaparity",
}

// TestParitySubcommandsReplaceTheirCommands holds each parity, one subtest by
// its name, to the command it replaced: the command is gone and the parity
// is reached by name through validate -parity, its help listing its own
// flags on a set named for it, without running a comparison. An unknown or
// missing name is refused with the parities listed.
func TestParitySubcommandsReplaceTheirCommands(t *testing.T) {
	t.Parallel()
	for name := range paritySubcommands {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			former, known := formerParityCommands[name]
			if !known {
				t.Fatalf("validate -parity %s names no command it replaced", name)
			}
			if _, err := os.Stat(filepath.Join("..", former)); !os.IsNotExist(err) {
				t.Errorf("cmd/%s still exists beside validate -parity %s (%v)", former, name, err)
			}
			var usage bytes.Buffer
			if err := runArgs(t.Context(), []string{"-parity", name, "-h"}, &usage); !errors.Is(err, flag.ErrHelp) ||
				!strings.Contains(usage.String(), "validate -parity "+name) || !strings.Contains(usage.String(), "-repo") && !strings.Contains(usage.String(), "-model") {
				t.Errorf("validate -parity %s -h = %v with usage %q, want its own flags", name, err, usage.String())
			}
		})
	}
	for _, args := range [][]string{{"-parity"}, {"-parity", "absent"}} {
		if err := runArgs(t.Context(), args, io.Discard); err == nil || !strings.Contains(err.Error(), "audio-oracle, sensenova, vqa") {
			t.Errorf("validate %v = %v, want the parities listed", args, err)
		}
	}
}
