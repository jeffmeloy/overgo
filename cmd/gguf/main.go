// Command gguf holds the model-file tools as subcommands, each with its own
// flags: gguf <subcommand> [flags] [arguments]. They convert Hugging Face
// checkpoints to GGUF, hash, inspect, merge, split or quantize GGUF files,
// and admit the audio files speech datasets are built from.
package main

import (
	"flag"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"

	"overgo/internal/clioptions"
)

// subcommands are the model-file tools, each once its own command.
var subcommands = map[string]func(flags *flag.FlagSet, args []string) error{
	"audio":          audioCommand,
	"convert":        convertCommand,
	"convert-gemma4": convertGemma4Command,
	"hash":           hashCommand,
	"inspect":        inspectCommand,
	"merge":          mergeCommand,
	"quantize":       quantizeCommand,
	"safetensors":    safetensorsCommand,
	"split":          splitCommand,
}

func main() {
	clioptions.MainNamed("gguf", func() error { return runSubcommand(os.Args[1:]) })
}

// runSubcommand dispatches args (the subcommand name, then its flags and
// arguments) to the named tool, parsing its flags on a set named for it.
func runSubcommand(args []string) error {
	names := strings.Join(slices.Sorted(maps.Keys(subcommands)), ", ")
	if len(args) == 0 {
		return fmt.Errorf("gguf needs a subcommand: %s", names)
	}
	command, known := subcommands[args[0]]
	if !known {
		return fmt.Errorf("gguf %q is not a subcommand; subcommands: %s", args[0], names)
	}
	return command(flag.NewFlagSet("gguf "+args[0], flag.ContinueOnError), args[1:])
}
