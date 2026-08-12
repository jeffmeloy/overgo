// plan: campaign dispatch bookkeeping (owner directive 2026-08-09: the turn
// is the plan). Reads docs/plan.json — the machine-readable open-work
// surface — and tracks progress:
//
//	plan -next                     print the top open action
//	plan -advance <item> <step>    mark a step done (item closes when all
//	                               steps are done; step "." closes the item)
//	plan -status                   one line per item
//
// Dispatch doctrine lives in skill.md: under the campaign directive a
// session works the plan continuously; the only stops are the ones a human
// must answer (user-stop, irreversible, external-prereq).
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/jsonfile"
)

const planPath = "docs/plan.json"

type step struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Status string `json:"status"`
}

type item struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Status string `json:"status"`
	Steps  []step `json:"steps"`
}

type plan struct {
	Campaign string `json:"campaign"`
	Doctrine string `json:"doctrine"`
	Items    []item `json:"items"`
}

func main() {
	next := flag.Bool("next", false, "print the top open action")
	status := flag.Bool("status", false, "one line per item")
	advance := flag.Bool("advance", false, "mark <item> <step> done")
	flag.Parse()
	if err := run(*next, *status, *advance, flag.Args()); err != nil {
		fmt.Fprintf(os.Stderr, "plan: %v\n", err)
		os.Exit(1)
	}
}

func run(next, status, advance bool, args []string) error {
	document, err := load()
	if err != nil {
		return err
	}
	switch {
	case advance:
		if len(args) != 2 {
			return errors.New("usage: plan -advance <item-id> <step-id|.>")
		}
		return advanceStep(document, args[0], args[1])
	case status:
		for _, entry := range document.Items {
			done, total := 0, len(entry.Steps)
			for _, s := range entry.Steps {
				if s.Status == "done" {
					done++
				}
			}
			fmt.Printf("%-22s %-6s %d/%d %s\n", entry.ID, entry.Status, done, total, entry.Title)
		}
		return nil
	case next:
		action, open := nextAction(document)
		if !open {
			fmt.Println("plan complete: every item is done")
			return nil
		}
		fmt.Println(action)
		return nil
	default:
		return errors.New("one of -next, -status, -advance is required")
	}
}

func load() (plan, error) {
	var document plan
	if err := jsonfile.Decode(filepath.FromSlash(planPath), &document); err != nil {
		return plan{}, fmt.Errorf("parse %s: %w", planPath, err)
	}
	return document, nil
}

func save(document plan) error {
	raw, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.FromSlash(planPath), append(raw, '\n'), 0o644)
}

// nextAction: first open step of the first open item; items without steps
// are themselves the action (opening the rung defines its steps).
func nextAction(document plan) (string, bool) {
	for _, entry := range document.Items {
		if entry.Status != "open" {
			continue
		}
		for _, s := range entry.Steps {
			if s.Status == "open" {
				return fmt.Sprintf("%s / %s: %s — %s", entry.ID, s.ID, entry.Title, s.Title), true
			}
		}
		return fmt.Sprintf("%s: %s — open the rung (rung export first, then define its steps)", entry.ID, entry.Title), true
	}
	return "", false
}

func advanceStep(document plan, itemID, stepID string) error {
	for i := range document.Items {
		if document.Items[i].ID != itemID {
			continue
		}
		if stepID == "." {
			document.Items[i].Status = "done"
			return finishAdvance(document, itemID, stepID)
		}
		for j := range document.Items[i].Steps {
			if document.Items[i].Steps[j].ID != stepID {
				continue
			}
			document.Items[i].Steps[j].Status = "done"
			allDone := true
			for _, s := range document.Items[i].Steps {
				allDone = allDone && s.Status == "done"
			}
			if allDone {
				document.Items[i].Status = "done"
			}
			return finishAdvance(document, itemID, stepID)
		}
		return fmt.Errorf("step %q not found in %q", stepID, itemID)
	}
	return fmt.Errorf("item %q not found", itemID)
}

func finishAdvance(document plan, itemID, stepID string) error {
	if err := save(document); err != nil {
		return err
	}
	action, open := nextAction(document)
	if !open {
		fmt.Printf("advanced %s/%s; PLAN COMPLETE\n", itemID, stepID)
		return nil
	}
	fmt.Printf("advanced %s/%s; next: %s\n", itemID, stepID, action)
	return nil
}
