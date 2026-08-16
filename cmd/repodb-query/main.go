package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("repodb-query", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "RepoDB root")
	kindText := flags.String("kind", "", "artifact kind")
	idText := flags.String("id", "", "artifact ID")
	alias := flags.String("alias", "", "exact alias")
	relationText := flags.String("relation", "", "lineage relation")
	followText := flags.String("follow", "none", "none, parents, children, or both")
	maxDepth := flags.Uint("max-depth", 16, "lineage traversal depth")
	limit := flags.Int("limit", 1_000, "maximum results per result class")
	from := flags.Uint64("from-sequence", 0, "first commit sequence")
	to := flags.Uint64("to-sequence", 0, "last commit sequence")
	jsonOutput := flags.Bool("json", false, "emit JSON")
	servable := flags.Bool("servable", false, "list models with an active inference recipe and on-disk presence (the discovery query)")
	generations := flags.Bool("generations", false, "list generation records with descendant depth derived from the committed graph")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || strings.TrimSpace(*repository) == "" {
		return errors.New("usage: repodb-query -repo <path> [filters]")
	}
	if *servable {
		return writeServable(output, *repository, *limit)
	}
	if *generations {
		return writeGenerations(output, *repository, *limit)
	}
	query := repodb.Query{
		Alias: *alias, MaxDepth: uint32(*maxDepth), MaxResults: *limit,
		FromSequence: *from, ToSequence: *to,
	}
	var err error
	if *kindText != "" {
		query.Kind, err = artifact.ParseKind(*kindText)
		if err != nil {
			return err
		}
	}
	if *idText != "" {
		id, parseErr := artifact.ParseID(*idText)
		if parseErr != nil {
			return parseErr
		}
		query.Artifact = &id
	}
	if *relationText != "" {
		query.Relation, err = artifact.ParseRelation(*relationText)
		if err != nil {
			return err
		}
	}
	query.Follow, err = parseFollow(*followText)
	if err != nil {
		return err
	}
	store, err := repodb.OpenReadOnly(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	result, err := store.Query(context.Background(), query)
	if err != nil {
		return err
	}
	if *jsonOutput {
		encoder := json.NewEncoder(output)
		encoder.SetIndent("", "  ")
		return encoder.Encode(result)
	}
	return writeText(output, result)
}

// writeServable renders the discovery predicate owned by internal/discovery.
func writeServable(output io.Writer, repository string, limit int) error {
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	entries, err := discovery.Servable(context.Background(), store, limit)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		fmt.Fprintf(output, "servable model=%s tier=%s recipe=%s present=%t location=%s\n",
			entry.Model, entry.Tier, entry.Recipe, entry.Present, entry.Location)
	}
	fmt.Fprintf(output, "%d servable model(s); honesty: every listed file hashes to its recorded component identity\n", len(entries))
	return nil
}

// writeGenerations lists committed generation records and derives each child's
// descendant-generation depth from the record graph: a model without a record
// is a depth-0 root, and a recorded child is one deeper than its deepest
// parent. Depth is computed here, never stored in a record.
func writeGenerations(output io.Writer, repository string, limit int) error {
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindEvidence, MaxResults: limit})
	if err != nil {
		return err
	}
	records := make(map[artifact.ID]runrecord.GenerationRecord)
	for _, descriptor := range result.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		record, err := runrecord.ParseGenerationRecord(content.Data)
		if err != nil {
			continue // other evidence document kinds share the store
		}
		records[record.Child] = record
	}
	var depthOf func(child artifact.ID, visiting map[artifact.ID]bool) int
	depthOf = func(child artifact.ID, visiting map[artifact.ID]bool) int {
		record, recorded := records[child]
		if !recorded || visiting[child] {
			return 0
		}
		visiting[child] = true
		defer delete(visiting, child)
		deepest := 0
		for _, parent := range record.Parents {
			if depth := depthOf(parent, visiting); depth > deepest {
				deepest = depth
			}
		}
		return 1 + deepest
	}
	children := make([]artifact.ID, 0, len(records))
	for child := range records {
		children = append(children, child)
	}
	slices.SortFunc(children, func(a, b artifact.ID) int { return strings.Compare(a.String(), b.String()) })
	for _, child := range children {
		record := records[child]
		fmt.Fprintf(output, "generation child=%s depth=%d outcome=%s parents=%d components=%d seeds=%d decision=%s\n",
			child, depthOf(child, map[artifact.ID]bool{}), record.Outcome,
			len(record.Parents), len(record.Components), len(record.Seeds), record.Decision)
	}
	fmt.Fprintf(output, "%d generation record(s); honesty: depth derives from committed generation records only; models without records are depth-0 roots\n", len(records))
	return nil
}

func parseFollow(value string) (repodb.FollowDirection, error) {
	switch value {
	case "none":
		return repodb.FollowNone, nil
	case "parents":
		return repodb.FollowParents, nil
	case "children":
		return repodb.FollowChildren, nil
	case "both":
		return repodb.FollowBoth, nil
	default:
		return repodb.FollowNone, fmt.Errorf("repodb-query: invalid follow direction %q", value)
	}
}

func writeText(output io.Writer, result repodb.QueryResult) error {
	if _, err := fmt.Fprintf(output, "head %s sequence %d\n", result.Head, result.Sequence); err != nil {
		return err
	}
	for _, descriptor := range result.Artifacts {
		if _, err := fmt.Fprintf(
			output, "artifact %s size=%d media=%q schema=%q\n",
			descriptor.ID, descriptor.Size, descriptor.MediaType, descriptor.Schema,
		); err != nil {
			return err
		}
	}
	for _, manifest := range result.Manifests {
		if _, err := fmt.Fprintf(output, "manifest %s components=%d\n", manifest.ID, len(manifest.Components)); err != nil {
			return err
		}
	}
	for _, alias := range result.Aliases {
		if _, err := fmt.Fprintf(output, "alias %s -> %s\n", alias.Name, alias.Target); err != nil {
			return err
		}
	}
	for _, edge := range result.Lineage {
		if _, err := fmt.Fprintf(output, "lineage %s -[%s]-> %s\n", edge.Child, edge.Relation, edge.Parent); err != nil {
			return err
		}
	}
	for _, commit := range result.Commits {
		if _, err := fmt.Fprintf(output, "commit %d %s key=%q\n", commit.Sequence, commit.ID, commit.Key); err != nil {
			return err
		}
	}
	if result.Truncated {
		_, err := fmt.Fprintln(output, "truncated true")
		return err
	}
	return nil
}
