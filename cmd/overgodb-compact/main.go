// overgodb-compact is retention's operator entry: it writes the live set of an
// store -- everything reachable from any alias through lineage and through
// identities embedded in retained document content -- into a fresh store,
// and reports exactly what it kept and what stayed behind. The source is
// never modified; swapping the compacted store into service and archiving
// the original is the operator's explicit, reversible act.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.MainNamed("overgodb-compact", run)
}

func run() error {
	repository := flag.String("repo", "overgodb-store", "source OvergoDB store")
	destination := flag.String("dest", "", "destination store root; must be empty or absent")
	flag.Parse()
	if strings.TrimSpace(*destination) == "" || flag.NArg() != 0 {
		return errors.New("usage: overgodb-compact -dest <new-store> [-repo <path>]")
	}
	source, err := overgodb.OpenReadOnly(*repository)
	if err != nil {
		return err
	}
	defer source.Close()
	report, err := overgodb.Compact(context.Background(), source, *destination, runrecord.GateLifecycleAtRest(source))
	if err != nil {
		return err
	}
	fmt.Println(report.String())
	fmt.Println("source untouched; swap stores and archive the original to take effect")
	return nil
}
