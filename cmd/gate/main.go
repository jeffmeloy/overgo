// Command gate lands one plan row through the small gate in internal/landing.
package main

import (
	"errors"
	"flag"

	"overgo/internal/clioptions"
)

func main() {
	clioptions.MainNamed("gate", run)
}

func run() error {
	messageFile := flag.String("message-file", "", "commit message file, or - for standard input (required; never -m: the shell eats backticks)")
	planRef := flag.String("plan", "", "item/step this commit serves, or several joined by commas that land together; each step must be open in the plan")
	preflight := flag.Bool("preflight", false, "run only the cheap stages (format, fix, vet, rules, the invariant tests, the row's verify) over the same tree and commit nothing")
	flag.Parse()
	if *planRef == "" || *messageFile == "" || flag.NArg() != 0 {
		return errors.New("usage: gate -plan <item>/<step> -message-file <file, or -> [-preflight]")
	}
	return land(*planRef, *messageFile, *preflight)
}
