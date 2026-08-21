// closure-scan: numeric-policy census and exact evidence publication.
package main

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
	"overgo/internal/repodb"
	"overgo/internal/strictjson"
)

type triageRow struct {
	Kind          closureledger.BindingKind `json:"kind"`
	Name          string                    `json:"name"`
	File          string                    `json:"file"`
	Scope         string                    `json:"scope"`
	Line          int                       `json:"line"`
	Tier          string                    `json:"tier"`
	Status        string                    `json:"status"`
	Understanding string                    `json:"understanding"`
	ClosurePath   string                    `json:"closure_path"`
	RerankTrigger string                    `json:"rerank_trigger"`
}

type triageFile struct {
	Rows []triageRow `json:"rows"`
}

type closureRequirements struct {
	classified, noStale, noUncatalogued, noModelFacts, zeroOpen bool
}

type testRequirements struct {
	noPolicyCopies, classifiedFixtures bool
}

func main() {
	triagePath := flag.String("triage", "", "triage JSON ({rows:[{kind,name,file,scope,line,tier,status,understanding,closure_path,rerank_trigger}]})")
	storePath := flag.String("store", "repodb-store", "RepoDB store directory (emit mode)")
	limit := flag.Int("limit", 40, "report mode: top-N candidates to print")
	raw := flag.Bool("raw", false, "rank repeated raw policy literals instead of declared constants")
	literals := flag.Bool("literals", false, "report classified production literals instead of declared constants")
	testLiterals := flag.Bool("test-literals", false, "report classified test literals and production overlaps")
	assumptions := flag.Bool("assumptions", false, "report syntax-derived distribution, geometry, and shape hints")
	census := flag.Bool("census", false, "report complete source denominators and consolidation pressure")
	format := flag.String("format", "text", "census format: text or json")
	publish := flag.Bool("publish", false, "publish census evidence to RepoDB")
	retireOrphans := flag.Bool("retire-orphans", false, "retire active bindings whose declarations were deleted")
	rebindUnchanged := flag.Bool("rebind-unchanged", false, "move unchanged decisions to current exact source bindings")
	checkScope := flag.String("check-scope", "", "comma-separated production package prefixes to validate")
	checkAll := flag.Bool("check-all", false, "validate every production package")
	checkTests := flag.Bool("check-tests", false, "validate test literal ownership")
	requireClassified := flag.Bool("require-classified", false, "require active closure evidence for every scoped constant")
	requireNoStale := flag.Bool("require-no-stale", false, "reject scoped closure binding drift")
	requireNoUncatalogued := flag.Bool("require-no-uncatalogued-production", false, "reject production policy without exact closure evidence")
	requireNoModelFacts := flag.Bool("require-no-model-facts", false, "reject unresolved model-runtime facts")
	requireNoPolicyCopies := flag.Bool("require-no-policy-copies", false, "reject tests that copy production policy")
	requireClassifiedFixtures := flag.Bool("require-classified-fixtures", false, "require an exact test-literal class")
	requireZeroOpen := flag.Bool("require-zero-open", false, "reject active derivation-blocked closure rows")
	flag.Parse()
	root, err := os.Getwd()
	if err != nil {
		fatal(err)
	}
	modes := 0
	for _, enabled := range []bool{
		*raw, *literals, *testLiterals, *assumptions, *census, *retireOrphans, *rebindUnchanged,
		*checkScope != "", *checkAll, *checkTests,
	} {
		if enabled {
			modes++
		}
	}
	if modes > 1 {
		fatal(fmt.Errorf("report modes are mutually exclusive"))
	}
	productionRequirements := closureRequirements{
		classified: *requireClassified, noStale: *requireNoStale,
		noUncatalogued: *requireNoUncatalogued, noModelFacts: *requireNoModelFacts, zeroOpen: *requireZeroOpen,
	}
	tests := testRequirements{noPolicyCopies: *requireNoPolicyCopies, classifiedFixtures: *requireClassifiedFixtures}
	if *checkScope != "" || *checkAll {
		if err := checkProductionClosures(root, *storePath, *checkScope, *checkAll, productionRequirements); err != nil {
			fatal(err)
		}
		if *requireNoPolicyCopies || *requireClassifiedFixtures {
			if err := checkTestAuthority(mustSnapshot(root), tests); err != nil {
				fatal(err)
			}
		}
		return
	}
	if *checkTests {
		if err := checkTestAuthority(mustSnapshot(root), tests); err != nil {
			fatal(err)
		}
		return
	}
	if *requireClassified || *requireNoStale || *requireNoUncatalogued || *requireNoModelFacts ||
		*requireNoPolicyCopies || *requireClassifiedFixtures || *requireZeroOpen {
		fatal(errors.New("closure requirements need -check-scope, -check-all, or -check-tests"))
	}
	if *census {
		snapshot := mustSnapshot(root)
		summary, err := closurescan.BuildCensus(snapshot)
		if err != nil {
			fatal(err)
		}
		switch *format {
		case "text":
			err = writeCensusText(os.Stdout, summary, *limit)
		case "json":
			err = clioptions.WritePrettyJSON(os.Stdout, summary)
		default:
			err = fmt.Errorf("unknown census format %q", *format)
		}
		if err != nil {
			fatal(err)
		}
		if *publish {
			store, err := repodb.Open(filepath.Join(root, *storePath))
			if err != nil {
				fatal(err)
			}
			evidence, err := publishCensusEvidence(context.Background(), store, snapshot, summary)
			closeErr := store.Close()
			if err != nil {
				fatal(err)
			}
			if closeErr != nil {
				fatal(closeErr)
			}
			fmt.Printf("published census evidence %s\n", evidence.ID)
		}
		return
	}
	if *retireOrphans {
		retired, err := retireOrphanAliases(root, *storePath, mustSnapshot(root))
		if err != nil {
			fatal(err)
		}
		fmt.Printf("retired %d orphan closure binding(s)\n", len(retired))
		return
	}
	if *rebindUnchanged {
		count, err := rebindUnchangedClosures(root, *storePath, mustSnapshot(root))
		if err != nil {
			fatal(err)
		}
		fmt.Printf("rebound %d unchanged closure document(s)\n", count)
		return
	}
	if *raw {
		ranked, err := closurescan.RepeatedPolicyLiterals(mustSnapshot(root))
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
	kinds := closurescan.CandidateConstants
	if *triagePath != "" {
		kinds = closurescan.CandidateAll
	}
	candidates, err := closurescan.ScanRoot(root, kinds)
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

func checkProductionClosures(root, storePath, scopeList string, all bool, requirements closureRequirements) error {
	snapshot := mustSnapshot(root)
	var prefixes []string
	if !all {
		var err error
		prefixes, err = closurePrefixes(scopeList)
		if err != nil {
			return err
		}
	}
	relatives := scopedSources(snapshot, prefixes)
	if len(relatives) == 0 {
		return errors.New("no production sources match closure scope")
	}
	candidates, err := closurescan.ScanSnapshot(snapshot, relatives, closurescan.CandidateAll)
	if err != nil {
		return err
	}
	store, err := repodb.OpenReadOnly(filepath.Join(root, storePath))
	if err != nil {
		return err
	}
	defer store.Close()
	documents, _, _, err := activeClosureDocuments(context.Background(), store)
	if err != nil {
		return err
	}
	if requirements.noStale {
		issues, err := closurescan.ValidateBindings(snapshot, scopedDocuments(documents, prefixes))
		if err != nil {
			return err
		}
		if len(issues) > 0 {
			return fmt.Errorf("%d scoped closure binding(s) stale; first=%s:%s", len(issues), issues[0].File, issues[0].Name)
		}
	}
	active := compileActiveClosures(documents)
	classified := 0
	for _, candidate := range candidates {
		document, found := activeCandidateClosure(active, candidate)
		if found {
			classified++
		}
		if requirements.classified && candidate.Kind == closureledger.BindingConstant && !found {
			return fmt.Errorf("unclassified scoped constant %s at %s:%d", candidate.Name, candidate.File, candidate.Line)
		}
		if requirements.noUncatalogued && candidate.Policy && !found {
			return fmt.Errorf("uncatalogued production policy %s at %s:%d", candidate.Name, candidate.File, candidate.Line)
		}
		if requirements.noModelFacts && modelAuthorityPackage(candidate.Package) && candidate.Policy &&
			(!found || document.Status == closureledger.StatusOpen) {
			return fmt.Errorf("unresolved model fact %s at %s:%d", candidate.Name, candidate.File, candidate.Line)
		}
	}
	if requirements.zeroOpen {
		for _, document := range documents {
			if document.Status == closureledger.StatusOpen {
				return fmt.Errorf("open closure row %s", document.Name)
			}
		}
	}
	fmt.Printf("closure-scan: scoped sites=%d classified=%d stale=0\n", len(candidates), classified)
	return nil
}

func compileActiveClosures(documents []closureledger.Document) map[string]closureledger.Document {
	active := make(map[string]closureledger.Document, len(documents))
	for _, document := range documents {
		for _, binding := range document.Bindings {
			key := (closurescan.Candidate{
				Kind: binding.Kind, File: binding.File, Scope: binding.Scope, Line: binding.Line, Name: binding.Name,
			}).DeclarationKey()
			active[key] = document
		}
	}
	return active
}

func activeCandidateClosure(active map[string]closureledger.Document, candidate closurescan.Candidate) (closureledger.Document, bool) {
	document, found := active[candidate.DeclarationKey()]
	if !found || !bytes.Equal(document.Value, candidate.ValueJSON()) {
		return closureledger.Document{}, false
	}
	binding, err := candidate.Binding()
	if err != nil || !slices.Contains(document.Bindings, binding) {
		return closureledger.Document{}, false
	}
	return document, true
}

func modelAuthorityPackage(pkg string) bool {
	for _, prefix := range []string{
		"internal/model", "internal/inference", "internal/projector",
		"internal/latentimage", "internal/latentvideo", "internal/speechsynth",
	} {
		if scopedPath(pkg, []string{prefix}) {
			return true
		}
	}
	return false
}

func checkTestAuthority(snapshot repoanalysis.SourceSnapshot, requirements testRequirements) error {
	sites, err := closurescan.CensusTestLiterals(snapshot)
	if err != nil {
		return err
	}
	var policyCopies int
	for _, site := range sites {
		if site.Class == closurescan.TestPolicyCopy {
			policyCopies++
		}
		if requirements.noPolicyCopies && site.Class == closurescan.TestPolicyCopy {
			return fmt.Errorf("test policy copy at %s:%d matches %s", site.File, site.Line, strings.Join(site.ProductionMatches, ","))
		}
		if requirements.classifiedFixtures && site.Class == "" {
			return fmt.Errorf("unclassified test literal at %s:%d", site.File, site.Line)
		}
	}
	fmt.Printf("closure-scan: test sites=%d policy_copies=%d\n", len(sites), policyCopies)
	return nil
}

func closurePrefixes(value string) ([]string, error) {
	var prefixes []string
	for _, raw := range strings.Split(value, ",") {
		prefix := strings.TrimSuffix(filepath.ToSlash(strings.TrimSpace(raw)), "/")
		if prefix == "" || filepath.IsAbs(prefix) || prefix == "." || prefix == ".." || strings.HasPrefix(prefix, "../") {
			return nil, fmt.Errorf("invalid closure scope %q", raw)
		}
		prefixes = append(prefixes, prefix)
	}
	slices.Sort(prefixes)
	return slices.Compact(prefixes), nil
}

func scopedSources(snapshot repoanalysis.SourceSnapshot, prefixes []string) []string {
	var relatives []string
	for _, source := range snapshot.Files {
		if !source.Test && (prefixes == nil || scopedPath(source.Path, prefixes)) {
			relatives = append(relatives, source.Path)
		}
	}
	return relatives
}

func scopedDocuments(documents []closureledger.Document, prefixes []string) []closureledger.Document {
	return slices.DeleteFunc(slices.Clone(documents), func(document closureledger.Document) bool {
		if prefixes == nil {
			return false
		}
		return !slices.ContainsFunc(document.Bindings, func(binding closureledger.SourceBinding) bool {
			return scopedPath(binding.File, prefixes)
		})
	})
}

func scopedPath(path string, prefixes []string) bool {
	return slices.ContainsFunc(prefixes, func(prefix string) bool {
		return path == prefix || strings.HasPrefix(path, prefix+"/")
	})
}

func retireOrphanAliases(root, storePath string, snapshot repoanalysis.SourceSnapshot) ([]artifact.AliasBinding, error) {
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateAll)
	if err != nil {
		return nil, err
	}
	current := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		current[candidate.DeclarationKey()] = struct{}{}
	}
	store, err := repodb.Open(filepath.Join(root, storePath))
	if err != nil {
		return nil, err
	}
	defer store.Close()
	result, err := store.Query(context.Background(), repodb.Query{
		Kind: artifact.KindEvidence, MediaType: closureledger.MediaType,
		Schema: closureledger.Schema, MaxResults: repodb.MaxQueryResults,
	})
	if err != nil {
		return nil, err
	}
	if result.Truncated {
		return nil, errors.New("closure-scan: active-ledger query truncated")
	}
	var retirements []artifact.AliasBinding
	for _, alias := range result.Aliases {
		if !closureledger.IsActiveAlias(alias.Name) {
			continue
		}
		content, found, err := store.Content(context.Background(), alias.Target)
		if err != nil || !found {
			return nil, errors.New("closure-scan: active closure content absent")
		}
		document, err := closureledger.Parse(content.Data)
		if err != nil {
			return nil, err
		}
		for _, binding := range document.Bindings {
			bindingAlias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return nil, err
			}
			if bindingAlias != alias.Name {
				continue
			}
			key := (closurescan.Candidate{
				Kind: binding.Kind, File: binding.File, Scope: binding.Scope, Line: binding.Line, Name: binding.Name,
			}).DeclarationKey()
			if _, exists := current[key]; !exists {
				previous := alias.Target
				retirements = append(retirements, artifact.AliasBinding{
					Name: alias.Name, Target: alias.Target, Previous: &previous, Remove: true,
				})
			}
			break
		}
	}
	if retirements == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(retirements)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(encoded)
	batch := artifact.Batch{Key: "closure-scan/retire/" + hex.EncodeToString(digest[:]), Aliases: retirements}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return nil, err
	}
	return retirements, nil
}

func publishCensusEvidence(ctx context.Context, store *repodb.Store, snapshot repoanalysis.SourceSnapshot, census closurescan.Census) (closurescan.CensusEvidence, error) {
	active, _, result, err := activeClosureDocuments(ctx, store)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	issues, err := closurescan.ValidateBindings(snapshot, active)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	evidence, err := closurescan.NewCensusEvidence(census, result.Head, result.Sequence, active, issues)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	previous, found, err := artifact.ResolveAlias(ctx, store, closurescan.CensusEvidenceAlias)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	var prior *artifact.ID
	if found {
		prior = &previous
	}
	batch, err := evidence.Batch(prior)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return closurescan.CensusEvidence{}, err
	}
	stored, found, err := closurescan.ReadCensusEvidence(ctx, store, evidence.ID)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	if !found {
		return closurescan.CensusEvidence{}, fmt.Errorf("published census evidence is absent")
	}
	return stored, nil
}

func rebindUnchangedClosures(root, storePath string, snapshot repoanalysis.SourceSnapshot) (count int, finalErr error) {
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateAll)
	if err != nil {
		return count, err
	}
	store, err := repodb.Open(filepath.Join(root, storePath))
	if err != nil {
		return count, err
	}
	defer store.Close()
	documents, aliases, _, err := activeClosureDocuments(context.Background(), store)
	if err != nil {
		return count, err
	}
	batch := artifact.Batch{}
	files := map[string]artifact.ID{}
	var rebound []closureledger.Document
	index := closurescan.CompileRebindIndex(candidates)
	for _, document := range documents {
		current, changed, err := index.Rebind(document)
		if err != nil {
			return count, err
		}
		if !changed {
			continue
		}
		currentAliases := map[string]bool{}
		for _, binding := range current.Bindings {
			if _, ok := files[binding.File]; !ok {
				id, descriptor, location, err := fileArtifact(root, binding.File)
				if err != nil || id != binding.Owner {
					return count, fmt.Errorf("rebind %s: inconsistent source owner", binding.Name)
				}
				files[binding.File] = id
				batch.Artifacts = append(batch.Artifacts, descriptor)
				batch.Locations = append(batch.Locations, location)
			}
			currentAlias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return count, err
			}
			currentAliases[currentAlias] = true
			active := artifact.AliasBinding{Name: currentAlias, Target: current.ID}
			if prior, ok := aliases[currentAlias]; ok {
				if prior != document.ID {
					return count, fmt.Errorf("rebind %s: current source alias collision", binding.Name)
				}
				active.Previous = &prior
			}
			batch.Aliases = append(batch.Aliases, active)
		}
		for _, binding := range document.Bindings {
			previousAlias, err := closureledger.ActiveAlias(binding)
			if err != nil || aliases[previousAlias] != document.ID {
				return count, fmt.Errorf("rebind %s: active source alias mismatch", binding.Name)
			}
			if !currentAliases[previousAlias] {
				prior := document.ID
				batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: previousAlias, Target: document.ID, Previous: &prior, Remove: true})
			}
		}
		content, err := current.Content()
		if err != nil {
			return count, err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, current.Lineage()...)
		rebound = append(rebound, current)
	}
	if rebound == nil {
		return count, nil
	}
	encoded, err := json.Marshal(batch)
	if err != nil {
		return count, err
	}
	digest := sha256.Sum256(encoded)
	batch.Key = "closure-scan/rebind/" + hex.EncodeToString(digest[:])
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return count, err
	}
	for _, document := range rebound {
		for _, binding := range document.Bindings {
			_, found, err := closureledger.ResolveActiveBinding(context.Background(), store, binding, document.Value)
			if err != nil {
				return count, fmt.Errorf("verify rebound binding %s: %w", binding.Name, err)
			}
			if !found {
				return count, fmt.Errorf("verify rebound binding %s: absent", binding.Name)
			}
		}
	}
	return len(rebound), nil
}

func activeClosureDocuments(ctx context.Context, store *repodb.Store) ([]closureledger.Document, map[string]artifact.ID, repodb.QueryResult, error) {
	result, err := store.Query(ctx, repodb.Query{MediaType: closureledger.MediaType, MaxResults: repodb.MaxQueryResults})
	if err != nil || result.Truncated {
		if err == nil {
			err = errors.New("closure-scan: active-ledger query truncated")
		}
		return nil, nil, repodb.QueryResult{}, err
	}
	documents := map[artifact.ID]closureledger.Document{}
	aliases := map[string]artifact.ID{}
	for _, alias := range result.Aliases {
		if !closureledger.IsActiveAlias(alias.Name) {
			continue
		}
		content, found, err := store.Content(ctx, alias.Target)
		if err != nil {
			return nil, nil, repodb.QueryResult{}, err
		}
		if !found {
			return nil, nil, repodb.QueryResult{}, errors.New("closure-scan: active closure content absent")
		}
		document, err := closureledger.Parse(content.Data)
		if err != nil {
			return nil, nil, repodb.QueryResult{}, err
		}
		documents[document.ID], aliases[alias.Name] = document, alias.Target
	}
	ids := slices.Collect(maps.Keys(documents))
	slices.SortFunc(ids, func(left, right artifact.ID) int { return cmp.Compare(left.String(), right.String()) })
	active := make([]closureledger.Document, len(ids))
	for index, id := range ids {
		active[index] = documents[id]
	}
	return active, aliases, result, nil
}

func writeCensusText(destination io.Writer, census closurescan.Census, limit int) error {
	counts := census.Counts
	if _, err := fmt.Fprintf(destination,
		"closure-scan census %s\nsource %s\nfiles production=%d test=%d\n"+
			"surfaces named=%d inline=%d assumptions=%d test_policy=%d\n"+
			"tests total=%d fixture=%d assertion=%d policy_copy=%d\n"+
			"repeated groups=%d sites=%d\n",
		census.Schema, census.Source, counts.ProductionFiles, counts.TestFiles,
		counts.NamedConstants, counts.InlineLiterals, counts.AssumptionHints, counts.TestPolicyCopies,
		counts.TestLiterals, counts.TestFixtures, counts.TestAssertions, counts.TestPolicyCopies,
		counts.RepeatedGroups, counts.RepeatedSites); err != nil {
		return err
	}
	if _, err := fmt.Fprintln(destination, "owners decision repeated package"); err != nil {
		return err
	}
	for index, owner := range census.Owners {
		if index >= limit {
			break
		}
		if _, err := fmt.Fprintf(destination, "%d %d %s\n", owner.DecisionSurfaces, owner.RepeatedSites, owner.Package); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintln(destination, "files decision repeated test source"); err != nil {
		return err
	}
	for index, file := range census.Files {
		if index >= limit {
			break
		}
		if _, err := fmt.Fprintf(destination, "%d %d %t %s\n", file.DecisionSurfaces, file.RepeatedGroups, file.Test, file.File); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintln(destination, "candidates score count package value"); err != nil {
		return err
	}
	for index, group := range census.Repeated {
		if index >= limit {
			break
		}
		if _, err := fmt.Fprintf(destination, "%d %d %s %s\n", group.Score, group.Count, group.Package, group.Value); err != nil {
			return err
		}
	}
	return nil
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
			Kind: row.Kind, File: row.File, Scope: row.Scope, Line: row.Line, Name: row.Name,
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
