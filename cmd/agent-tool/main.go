// Command agent-tool publishes and audits the agent tool manual
// catalog: UTCP-style tool declarations become store authority under
// registered aliases, and orchestration resolves tools only from there.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/agenttool"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
)

func main() {
	clioptions.MainNamed("agent-tool", func() error { return run(os.Args[1:], os.Stdout) })
}

type manualFile struct {
	Manuals []agenttool.Manual `json:"manuals"`
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("agent-tool", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB root")
	manualsPath := flags.String("manuals", "", "JSON manual declarations ({manuals:[...]})")
	inspect := flags.Bool("inspect", false, "audit the declared manuals against the store without publishing")
	resolve := flags.String("resolve", "", "resolve one registered tool manual by name")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("usage: agent-tool [-repo <path>] -manuals <file> [-inspect] | -resolve <name>")
	}
	root := strings.TrimSpace(*repository)
	if root == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		root = roots.Store
	}
	if name := strings.TrimSpace(*resolve); name != "" {
		store, err := overgodb.OpenReadOnly(root)
		if err != nil {
			return err
		}
		defer store.Close()
		manual, err := agenttool.ResolveRegisteredManual(context.Background(), store, name)
		if err != nil {
			return err
		}
		return clioptions.WritePrettyJSON(output, manual)
	}
	manuals, err := loadManuals(*manualsPath)
	if err != nil {
		return err
	}
	if *inspect {
		store, err := overgodb.OpenReadOnly(root)
		if err != nil {
			return err
		}
		defer store.Close()
		coverage, err := agenttool.InspectManualCatalog(context.Background(), store, manuals)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "tools registered=%d published=%d complete=%t\n",
			coverage.Registered, coverage.Published, coverage.Complete)
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	publication, err := agenttool.PublishManualCatalog(context.Background(), store, manuals)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "tools registered=%d published=%d changed=%t commit=%s\n",
		publication.Coverage.Registered, publication.Coverage.Published,
		publication.Changed, publication.Commit)
	return err
}

func loadManuals(path string) ([]agenttool.Manual, error) {
	if strings.TrimSpace(path) == "" {
		return nil, errors.New("agent-tool: -manuals file is required")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var declared manualFile
	if err := strictjson.DecodeBytes(raw, &declared); err != nil {
		return nil, fmt.Errorf("agent-tool: decode manuals: %w", err)
	}
	manuals := make([]agenttool.Manual, len(declared.Manuals))
	for index, declaration := range declared.Manuals {
		manual, err := agenttool.NewManual(declaration)
		if err != nil {
			return nil, err
		}
		manuals[index] = manual
	}
	return manuals, nil
}
