// validate selects and runs the minimal sufficient end-to-end validation of the
// whole system for one commit: it accounts for every registered model, derives
// the appropriate validation for each declared function, and runs only the
// validations whose surface the commit's changes moved, reusing every cell that
// stays pinned. The default dry run computes and prints the plan without loading
// a model or a device; -run executes the selected cells.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/gitauthority"
	"overgo/internal/longform"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

// catalogLimit bounds the registered-model listing; a truncated denominator is
// refused rather than read as complete.
const catalogLimit = 4096

func main() { clioptions.MainNamed("validate", run) }

func run() error { return runArgs(context.Background(), os.Args[1:], os.Stdout) }

func runArgs(ctx context.Context, args []string, output io.Writer) error {
	flags := flag.NewFlagSet("validate", flag.ContinueOnError)
	root := flags.String("root", ".", "repository root: the commit range and surfaces are read from it")
	repository := flags.String("repo", "", "OvergoDB store; empty resolves through the data-root contract")
	baseline := flags.String("baseline", "HEAD~1", "commit to diff against; the change set is baseline..HEAD")
	execute := flags.Bool("run", false, "execute the selected validations (default: dry run prints the plan only)")
	inventory := flags.String("inventory", "", "comma-separated model roots: report each directory beneath them as validated, registered, unregistered or not a model, with the commands that would advance it; loads no model")
	command := clioptions.Command{
		Name:     "validate",
		Purpose:  "select the minimal sufficient end-to-end validation for a commit and, with -run, execute it",
		Audience: "the master-lead session and its unattended driver validating the whole system per commit",
		Constraints: []string{
			"the default dry run loads no model and opens no device; it prints the plan as JSON",
			"a cell runs only when the commit's changes move its surface or it has no accepted evidence",
			"every registered model is accounted for; each available model is dispatched by its declared functions",
			"-inventory starts from the directories on disk, so a model nothing has registered is reported; it ties a converted model to its directory by the content of the config files the conversion recorded, never by a name",
			"help opens no store and reads nothing",
		},
	}
	if handled, err := command.ParseForHelp(flags, args, output); err != nil || handled {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("validate: unexpected arguments %v", flags.Args())
	}

	head, err := revParse(ctx, *root, "HEAD")
	if err != nil {
		return err
	}
	base, err := revParse(ctx, *root, *baseline)
	if err != nil {
		return err
	}
	changed, err := changedPaths(ctx, *root, base, head)
	if err != nil {
		return err
	}
	affected, err := surfaceAffected(ctx, *root, changed)
	if err != nil {
		return err
	}

	storeRoot, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		return err
	}
	defer store.Close()
	registered, err := registeredCells(ctx, store)
	if err != nil {
		return err
	}
	if *inventory != "" {
		directories, err := modelDirectories(strings.Split(*inventory, ","))
		if err != nil {
			return err
		}
		converted, err := convertedSources(ctx, store)
		if err != nil {
			return err
		}
		// The surface the code is at now: a guard record is current only when
		// it ran on this one.
		surface, err := longform.Surface(ctx, *root)
		if err != nil {
			return err
		}
		guard := func(location string) error { return longform.Admit(ctx, store, location, surface) }
		return clioptions.WritePrettyJSON(output, buildInventory(directories, registered, converted, guard))
	}

	plan := SelectValidation(Inputs{Baseline: base, Head: head, ChangedPaths: changed, AffectedSurfaces: affected, Registered: registered})
	if *execute {
		return executeRun(ctx, output, *root, plan)
	}
	return clioptions.WritePrettyJSON(output, plan)
}

// registeredCells enumerates every registered model and derives its validation
// obligations from its declared functions, generically: a model with no
// activation, or absent bytes, is one registered-inactive cell; each activated
// function maps to the appropriate validation for its kind, keyed to the surface
// that pins its evidence. No per-model knowledge participates.
func registeredCells(ctx context.Context, store *overgodb.Store) ([]ModelValidation, error) {
	memo := discovery.LoadMemo(ctx, store)
	entries, truncated, err := discovery.RegisteredCatalog(ctx, store, catalogLimit, memo)
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, fmt.Errorf("validate: registered catalog exceeds listing bound %d; raise the bound", catalogLimit)
	}
	var cells []ModelValidation
	for _, entry := range entries {
		name := ""
		if entry.Location != "" {
			name = filepath.Base(entry.Location)
		}
		if !entry.Present || len(entry.Capabilities) == 0 {
			cells = append(cells, ModelValidation{Model: entry.Model, ModelName: name, Location: entry.Location})
			continue
		}
		modality := declaredModality(ctx, store, entry.Model)
		for _, capability := range entry.Capabilities {
			validations := kindValidation(capability.Task)
			if len(validations) == 0 {
				cells = append(cells, ModelValidation{Model: entry.Model, ModelName: name, Location: entry.Location, Kind: capability.Task, Modality: modality})
				continue
			}
			for _, validation := range validations {
				cell := ModelValidation{
					Model: entry.Model, ModelName: name, Location: entry.Location,
					Kind: capability.Task, Modality: modality,
					Validation: validation.Validation, Surface: validation.Surface, Stale: capability.Stale,
				}
				// An active, non-stale activation carries accepted evidence pinned
				// by its recipe; a stale one has none and must re-acquire. Binding
				// the exact accepted record and its measured cost is a later cell.
				if capability.Stale == "" {
					cell.Evidence = capability.Recipe
				}
				cells = append(cells, cell)
			}
		}
	}
	return cells, nil
}

// declaredModality joins a model's declared evaluation domains for display; an
// undeclared or unreadable model contributes no modality label.
func declaredModality(ctx context.Context, store *overgodb.Store, model artifact.ID) string {
	domains, declared, err := modelartifact.EvalDomains(ctx, store, model)
	if err != nil || !declared || len(domains) == 0 {
		return ""
	}
	return strings.Join(domains, ",")
}

// revParse resolves a git revision to its object id through the read-only git owner.
func revParse(ctx context.Context, root, revision string) (string, error) {
	out, err := gitauthority.Query(ctx, root, "rev-parse", revision)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// changedPaths lists the repository-relative paths that differ across base..head.
func changedPaths(ctx context.Context, root, base, head string) ([]string, error) {
	out, err := gitauthority.Query(ctx, root, "diff", "--name-only", base+".."+head)
	if err != nil {
		return nil, err
	}
	var paths []string
	for line := range strings.SplitSeq(strings.TrimSpace(string(out)), "\n") {
		if trimmed := strings.TrimSpace(line); trimmed != "" {
			paths = append(paths, trimmed)
		}
	}
	return paths, nil
}
