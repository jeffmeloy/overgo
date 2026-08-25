// Command agent-tool publishes and audits the agent tool manual
// catalog: UTCP-style tool declarations become store authority under
// registered aliases, and orchestration resolves tools only from there.
package main

import (
	"context"
	"encoding/json"
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
	argvAllow := flags.String("argv-allow", "", "comma-separated programs to publish as the durable argv policy before the manuals; the committed policy is what publication and invocation enforce")
	inspect := flags.Bool("inspect", false, "audit the declared manuals against the store without publishing")
	resolve := flags.String("resolve", "", "resolve one registered tool manual by name")
	invoke := flags.String("invoke", "", "invoke one registered tool manual by name over its declared transport")
	arguments := flags.String("arguments", "{}", "strict JSON object of tool arguments for -invoke")
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
	if name := strings.TrimSpace(*invoke); name != "" {
		store, err := overgodb.OpenReadOnly(root)
		if err != nil {
			return err
		}
		defer store.Close()
		ctx := context.Background()
		manual, err := agenttool.ResolveRegisteredManual(ctx, store, name)
		if err != nil {
			return err
		}
		executor := agenttool.NewOperatorExecutor()
		if err := agenttool.RegisterStandardBuiltins(executor, store); err != nil {
			return err
		}
		result, err := executor.Invoke(ctx, manual, json.RawMessage(*arguments))
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "%s\n", result)
		return err
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
	if strings.TrimSpace(*argvAllow) != "" {
		var programs []string
		for _, program := range strings.Split(*argvAllow, ",") {
			if trimmed := strings.TrimSpace(program); trimmed != "" {
				programs = append(programs, trimmed)
			}
		}
		policy, err := agenttool.PublishArgvPolicy(context.Background(), store, programs)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "argv policy %s programs=%d\n", policy.ID, len(policy.Programs))
	}
	publication, err := agenttool.PublishManualCatalog(context.Background(), store, manuals)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "tools registered=%d published=%d changed=%t commit=%s\n",
		publication.Coverage.Registered, publication.Coverage.Published,
		publication.Changed, publication.Commit)
	return err
}

// loadManuals returns the standard store-inspection manuals plus any
// declared in the optional file; declarations never shadow a standard
// name, so the inspection ground stays canonical.
func loadManuals(path string) ([]agenttool.Manual, error) {
	manuals, err := agenttool.StandardManuals()
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(path) == "" {
		return manuals, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var declared manualFile
	if err := strictjson.DecodeBytes(raw, &declared); err != nil {
		return nil, fmt.Errorf("agent-tool: decode manuals: %w", err)
	}
	standard := map[string]bool{}
	for _, manual := range manuals {
		standard[manual.Name] = true
	}
	for _, declaration := range declared.Manuals {
		manual, err := agenttool.NewManual(declaration)
		if err != nil {
			return nil, err
		}
		if standard[manual.Name] {
			return nil, fmt.Errorf("agent-tool: %q shadows a standard manual", manual.Name)
		}
		manuals = append(manuals, manual)
	}
	return manuals, nil
}
