// Command overgodb holds the store tools as subcommands, each with its own
// flags: overgodb <subcommand> [flags]. They back up, check, import,
// precheck, query, rebuild and repair an OvergoDB store, and count its
// checkpoint by record family.
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

// subcommands are the store tools, each once its own command.
var subcommands = map[string]func(flags *flag.FlagSet, args []string) error{
	"backup": func(flags *flag.FlagSet, args []string) error { return backup(flags, args, os.Stdout) },
	"check":  checkStore,
	"checkpoint-families": func(flags *flag.FlagSet, args []string) error {
		return checkpointFamilies(flags, args, os.Stdout)
	},
	"import": func(flags *flag.FlagSet, args []string) error {
		return importStream(flags, args, os.Stdin, os.Stdout)
	},
	"precheck": precheckStore,
	"query":    func(flags *flag.FlagSet, args []string) error { return query(flags, args, os.Stdout) },
	"rebuild":  rebuild,
	"repair":   repair,
}

func main() {
	clioptions.MainNamed("overgodb", func() error { return runSubcommand(os.Args[1:]) })
}

// runSubcommand dispatches args (the subcommand name, then its flags) to the
// named tool, parsing its flags on a set named for it.
func runSubcommand(args []string) error {
	names := strings.Join(slices.Sorted(maps.Keys(subcommands)), ", ")
	if len(args) == 0 {
		return fmt.Errorf("overgodb needs a subcommand: %s", names)
	}
	command, known := subcommands[args[0]]
	if !known {
		return fmt.Errorf("overgodb %q is not a subcommand; subcommands: %s", args[0], names)
	}
	return command(flag.NewFlagSet("overgodb "+args[0], flag.ContinueOnError), args[1:])
}
