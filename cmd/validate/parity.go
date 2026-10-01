package main

import (
	"flag"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
)

// paritySubcommands are the reference comparisons validate -parity runs by
// name, each once its own command: the external audio-oracle election and
// the SenseNova and RxBrain VQA ladders against their pinned fixtures.
var paritySubcommands = map[string]func(flags *flag.FlagSet, args []string) error{
	"audio-oracle": audioOracleParity,
	"sensenova":    senseNovaParity,
	"vqa":          vqaParity,
}

// runParity dispatches args (the parity name, then its flags) to the named
// comparison, parsing its flags on a set named for it whose usage goes to
// output.
func runParity(args []string, output io.Writer) error {
	names := strings.Join(slices.Sorted(maps.Keys(paritySubcommands)), ", ")
	if len(args) == 0 {
		return fmt.Errorf("validate -parity needs a name: %s", names)
	}
	command, known := paritySubcommands[args[0]]
	if !known {
		return fmt.Errorf("validate -parity %q is not a parity; parities: %s", args[0], names)
	}
	flags := flag.NewFlagSet("validate -parity "+args[0], flag.ContinueOnError)
	flags.SetOutput(output)
	return command(flags, args[1:])
}
