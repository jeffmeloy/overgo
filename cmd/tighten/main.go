// tighten applies source reductions whose full transformation is mechanically proven.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"overgo/internal/clioptions"
)

func main() {
	clioptions.MainNamed("tighten", run)
}

func run() error {
	root := flag.String("root", ".", "repository root")
	proposalID := flag.String("apply", "", "exact proposal id to apply and verify")
	flag.Parse()
	if *proposalID != "" {
		proposal, err := apply(*root, *proposalID)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "tighten: migrated %d calls and deleted %s; removed %d wrapper AST nodes\n", proposal.Calls, proposal.Wrapper, proposal.RemovedNodes)
		return nil
	}
	proposals, err := discover(*root)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	return encoder.Encode(proposals)
}
