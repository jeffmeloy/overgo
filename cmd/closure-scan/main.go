// closure-scan: numeric-policy census and exact evidence publication.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
	"overgo/internal/repodb"
	"overgo/internal/strictjson"
)

type triageRow struct {
	Name          string `json:"name"`
	File          string `json:"file"`
	Scope         string `json:"scope"`
	Line          int    `json:"line"`
	Tier          string `json:"tier"`
	Status        string `json:"status"`
	Understanding string `json:"understanding"`
	ClosurePath   string `json:"closure_path"`
	RerankTrigger string `json:"rerank_trigger"`
}

type triageFile struct {
	Rows []triageRow `json:"rows"`
}

func main() {
	triagePath := flag.String("triage", "", "triage JSON ({rows:[{name,file,tier,status,closure_path,rerank_trigger}]}); emits closure documents to the store")
	storePath := flag.String("store", "repodb-store", "RepoDB store directory (emit mode)")
	limit := flag.Int("limit", 40, "report mode: top-N candidates to print")
	raw := flag.Bool("raw", false, "rank repeated raw policy literals instead of declared constants")
	literals := flag.Bool("literals", false, "report classified production literals instead of declared constants")
	testLiterals := flag.Bool("test-literals", false, "report classified test literals and production overlaps")
	assumptions := flag.Bool("assumptions", false, "report syntax-derived distribution, geometry, and shape hints")
	flag.Parse()
	root, err := os.Getwd()
	if err != nil {
		fatal(err)
	}
	modes := 0
	for _, enabled := range []bool{*raw, *literals, *testLiterals, *assumptions} {
		if enabled {
			modes++
		}
	}
	if modes > 1 {
		fatal(fmt.Errorf("-raw, -literals, -test-literals, and -assumptions are mutually exclusive"))
	}
	if *raw {
		ranked, err := closurescan.RankRawPolicyLiterals(mustSnapshot(root))
		if err != nil {
			fatal(err)
		}
		reportRaw(ranked, *limit)
		return
	}
	if *literals {
		sites, err := closurescan.CensusLiterals(mustSnapshot(root), nil)
		if err != nil {
			fatal(err)
		}
		reportLiterals(sites, *limit)
		return
	}
	if *testLiterals {
		sites, err := closurescan.CensusTestLiterals(mustSnapshot(root))
		if err != nil {
			fatal(err)
		}
		reportTestLiterals(sites, *limit)
		return
	}
	if *assumptions {
		hints, err := closurescan.CensusAssumptions(mustSnapshot(root), nil)
		if err != nil {
			fatal(err)
		}
		reportAssumptions(hints, *limit)
		return
	}
	candidates, err := closurescan.ScanRoot(root)
	if err != nil {
		fatal(err)
	}
	if *triagePath == "" {
		report(candidates, *limit)
		return
	}
	if err := emit(root, *storePath, *triagePath, candidates); err != nil {
		fatal(err)
	}
}

func reportLiterals(sites []closurescan.LiteralSite, limit int) {
	fmt.Printf("closure-scan: %d classified production numeric literals (named constants excluded)\n", len(sites))
	fmt.Printf("%-18s %-16s %-24s %s\n", "context", "value", "scope", "source")
	for index, site := range sites {
		if index >= limit {
			fmt.Printf("... %d more (raise -limit)\n", len(sites)-limit)
			break
		}
		fmt.Printf("%-18s %-16s %-24s %s:%d\n", site.Context, site.Value, site.Scope, site.File, site.Line)
	}
}

func reportTestLiterals(sites []closurescan.TestLiteralSite, limit int) {
	counts := map[closurescan.TestLiteralClass]int{}
	for _, site := range sites {
		counts[site.Class]++
	}
	fmt.Printf("closure-scan: %d classified test literals (fixture=%d assertion=%d policy_copy=%d)\n",
		len(sites), counts[closurescan.TestFixture], counts[closurescan.TestAssertion], counts[closurescan.TestPolicyCopy])
	fmt.Printf("%-14s %-18s %-16s %-24s %s\n", "class", "context", "value", "scope", "source")
	for index, site := range sites {
		if index >= limit {
			fmt.Printf("... %d more (raise -limit)\n", len(sites)-limit)
			break
		}
		fmt.Printf("%-14s %-18s %-16s %-24s %s:%d\n",
			site.Class, site.Context, site.Value, site.Scope, site.File, site.Line)
	}
}

func reportAssumptions(hints []closurescan.AssumptionHint, limit int) {
	fmt.Printf("closure-scan: %d syntax-derived assumption hints\n", len(hints))
	fmt.Printf("%-22s %-24s %s\n", "kind", "scope", "source")
	for index, hint := range hints {
		if index >= limit {
			fmt.Printf("... %d more (raise -limit)\n", len(hints)-limit)
			break
		}
		fmt.Printf("%-22s %-24s %s:%d\n", hint.Kind, hint.Scope, hint.File, hint.Line)
	}
}

func mustSnapshot(root string) repoanalysis.SourceSnapshot {
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		fatal(err)
	}
	return snapshot
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "closure-scan: %v\n", err)
	os.Exit(1)
}

func report(candidates []closurescan.Candidate, limit int) {
	fmt.Printf("closure-scan: %d numeric constants (iota enums, tests, generated excluded)\n", len(candidates))
	fmt.Printf("%-6s %-44s %-16s %-24s %s\n", "score", "const", "value", "scope", "source")
	for index, row := range candidates {
		if index >= limit {
			fmt.Printf("... %d more (raise -limit)\n", len(candidates)-limit)
			break
		}
		fmt.Printf("%-6d %-44s %-16s %-24s %s:%d\n", row.Score, row.Name, row.Value, row.Scope, row.File, row.Line)
	}
}

func reportRaw(candidates []closurescan.RawPolicyLiteral, limit int) {
	fmt.Printf("closure-scan: %d repeated raw policy candidates (tests, generated, structural math excluded)\n", len(candidates))
	fmt.Printf("%-6s %-8s %-24s %-28s %s\n", "score", "count", "value", "package", "functions")
	for index, row := range candidates {
		if index >= limit {
			fmt.Printf("... %d more (raise -limit)\n", len(candidates)-limit)
			break
		}
		owners := row.Functions
		if len(owners) > 5 {
			owners = append(append([]string(nil), owners[:5]...), fmt.Sprintf("+%d", len(row.Functions)-5))
		}
		fmt.Printf("%-6d %-8d %-24s %-28s %s\n", row.Score, row.Count, row.Value, row.Package, strings.Join(owners, ","))
	}
}

func emit(root, storePath, triagePath string, candidates []closurescan.Candidate) error {
	raw, err := os.ReadFile(triagePath)
	if err != nil {
		return err
	}
	var triage triageFile
	if err := strictjson.DecodeBytes(raw, &triage); err != nil {
		return fmt.Errorf("triage file: %w", err)
	}
	if len(triage.Rows) == 0 {
		return fmt.Errorf("triage file has no rows")
	}
	byKey := map[string]closurescan.Candidate{}
	for _, row := range candidates {
		byKey[row.DeclarationKey()] = row
	}
	store, err := repodb.Open(filepath.Join(root, storePath))
	if err != nil {
		return err
	}
	defer store.Close()
	batch := artifact.Batch{}
	fileIDs := map[string]artifact.ID{}
	type publishedBinding struct {
		binding closureledger.SourceBinding
		value   json.RawMessage
	}
	published := make([]publishedBinding, 0, len(triage.Rows))
	for _, row := range triage.Rows {
		found, ok := byKey[(closurescan.Candidate{
			File: row.File, Scope: row.Scope, Line: row.Line, Name: row.Name,
		}).DeclarationKey()]
		if !ok {
			return fmt.Errorf("triage row %s not found by scan in %s (stale triage?)", row.Name, row.File)
		}
		fileID, ok := fileIDs[row.File]
		if !ok {
			id, descriptor, location, err := fileArtifact(root, row.File)
			if err != nil {
				return err
			}
			batch.Artifacts = append(batch.Artifacts, descriptor)
			batch.Locations = append(batch.Locations, location)
			fileIDs[row.File] = id
			fileID = id
		}
		valueJSON := found.ValueJSON()
		binding, err := found.Binding()
		if err != nil || binding.Owner != fileID {
			return fmt.Errorf("row %s: inconsistent source owner", row.Name)
		}
		document, err := closureledger.New(
			row.Name, valueJSON,
			closureledger.Tier(row.Tier), closureledger.Status(row.Status),
			row.Understanding, []closureledger.SourceBinding{binding},
			row.ClosurePath, row.RerankTrigger, fileID,
		)
		if err != nil {
			return fmt.Errorf("row %s: %w", row.Name, err)
		}
		content, err := document.Content()
		if err != nil {
			return err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, document.Lineage()...)
		alias, err := closureledger.ActiveAlias(binding)
		if err != nil {
			return err
		}
		active := artifact.AliasBinding{Name: alias, Target: document.ID}
		if previous, exists, err := artifact.ResolveAlias(context.Background(), store, alias); err != nil {
			return err
		} else if exists {
			active.Previous = &previous
		}
		batch.Aliases = append(batch.Aliases, active)
		published = append(published, publishedBinding{binding: binding, value: valueJSON})
	}
	encoded, err := json.Marshal(batch)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(encoded)
	batch.Key = "closure-scan/" + hex.EncodeToString(digest[:])
	commit, err := store.Commit(context.Background(), batch)
	if err != nil {
		return err
	}
	for _, row := range published {
		_, found, err := closureledger.ResolveActiveBinding(context.Background(), store, row.binding, row.value)
		if err != nil {
			return fmt.Errorf("verify active binding %s: %w", row.binding.Name, err)
		}
		if !found {
			return fmt.Errorf("verify active binding %s: absent", row.binding.Name)
		}
	}
	fmt.Printf("emitted %d closure documents (%d owner files) commit %x\n",
		len(triage.Rows), len(fileIDs), commit[:8])
	return nil
}

func fileArtifact(root, relative string) (artifact.ID, artifact.Descriptor, artifact.LocationEvent, error) {
	resolved := filepath.Join(root, filepath.FromSlash(relative))
	file, err := os.Open(resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	defer file.Close()
	id, size, err := artifact.Identify(artifact.KindFile, file)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	absolute, err := filepath.Abs(resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	return id, artifact.Descriptor{ID: id, Size: size}, artifact.LocationEvent{
		Location: artifact.Location{Artifact: id, Kind: artifact.LocationFile, Value: absolute},
		Action:   artifact.LocationAdd,
	}, nil
}
