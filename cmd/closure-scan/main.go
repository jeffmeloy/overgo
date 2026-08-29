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
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
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
	storePath := flag.String("store", "overgodb-store", "OvergoDB store directory (emit mode)")
	limit := flag.Int("limit", 40, "report mode: top-N candidates to print")
	literals := flag.Bool("literals", false, "report classified production literals instead of declared constants")
	testLiterals := flag.Bool("test-literals", false, "report classified test literals and production overlaps")
	assumptions := flag.Bool("assumptions", false, "report syntax-derived distribution, geometry, and shape hints")
	census := flag.Bool("census", false, "report complete source denominators and consolidation pressure")
	format := flag.String("format", "text", "census format: text or json")
	publish := flag.Bool("publish", false, "publish census evidence to OvergoDB")
	importStore := flag.String("import-store", "", "import and rebind matching active decisions from another OvergoDB store")
	reviewCallsites := flag.Bool("review-callsites", false, "with -import-store: accept reviewed callsite drift")
	recoverRetired := flag.Bool("recover-retired", false, "with -import-store: restore the newest matching retired decision only when its current alias is uncovered")
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
	if (*reviewCallsites || *recoverRetired) && *importStore == "" {
		fatal(errors.New("-review-callsites and -recover-retired require -import-store"))
	}
	root, err := os.Getwd()
	if err != nil {
		fatal(err)
	}
	modes := 0
	for _, enabled := range []bool{
		*literals, *testLiterals, *assumptions, *census, *importStore != "",
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
			err = writeCensusText(os.Stdout, summary)
		case "json":
			err = clioptions.WritePrettyJSON(os.Stdout, summary)
		default:
			err = fmt.Errorf("unknown census format %q", *format)
		}
		if err != nil {
			fatal(err)
		}
		if *publish {
			store, err := overgodb.Open(filepath.Join(root, *storePath))
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
	if *importStore != "" {
		count, unmatched, first, err := importClosureDocuments(
			root, *storePath, *importStore, mustSnapshot(root), *reviewCallsites, *recoverRetired,
		)
		if err != nil {
			fatal(err)
		}
		fmt.Printf("imported %d closure document(s), unmatched=%d first=%s\n", count, unmatched, first)
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
	store, err := overgodb.OpenReadOnly(filepath.Join(root, storePath))
	if err != nil {
		return err
	}
	defer store.Close()
	documents, _, err := activeClosureDocuments(context.Background(), store)
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
				Kind: binding.Kind, Package: binding.Package, File: binding.File, Scope: binding.Scope,
				Line: binding.Line, Name: binding.Name, StructuralID: binding.StructuralID,
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
	if err != nil || !slices.ContainsFunc(document.Bindings, func(current closureledger.SourceBinding) bool {
		return current.StructuralID == binding.StructuralID && current.SourceID == binding.SourceID &&
			current.CallsiteID == binding.CallsiteID && current.Expression == binding.Expression
	}) {
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

func publishCensusEvidence(ctx context.Context, store *overgodb.Store, snapshot repoanalysis.SourceSnapshot, census closurescan.Census) (closurescan.CensusEvidence, error) {
	active, _, err := activeClosureDocuments(ctx, store)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	issues, err := closurescan.ValidateBindings(snapshot, active)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	head, sequence := store.Head()
	evidence, err := closurescan.NewCensusEvidence(census, head, sequence, active, issues)
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

func importClosureDocuments(
	root, storePath, sourcePath string,
	snapshot repoanalysis.SourceSnapshot,
	reviewCallsites, recoverRetired bool,
) (count, unmatched int, first string, finalErr error) {
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateAll)
	if err != nil {
		return count, unmatched, first, err
	}
	destinationPath := filepath.Join(root, storePath)
	sourceInfo, sourceErr := os.Stat(sourcePath)
	destinationInfo, destinationErr := os.Stat(destinationPath)
	sameStore := sourceErr == nil && destinationErr == nil && os.SameFile(sourceInfo, destinationInfo)
	source, err := overgodb.OpenReadOnly(sourcePath)
	if err != nil {
		return count, unmatched, first, err
	}
	var target *overgodb.Store
	defer func() {
		if source != nil {
			finalErr = errors.Join(finalErr, source.Close())
		}
		if target != nil && target != source {
			finalErr = errors.Join(finalErr, target.Close())
		}
	}()
	documents, sourceAliases, err := activeClosureDocuments(context.Background(), source)
	if err != nil {
		return count, unmatched, first, err
	}
	if recoverRetired {
		historical, err := retiredClosureDocuments(context.Background(), source, documents)
		if err != nil {
			return count, unmatched, first, err
		}
		documents = append(documents, historical...)
	}
	targetDocuments, targetAliases := documents, sourceAliases
	if sameStore {
		target = source
	} else {
		target, err = overgodb.Open(destinationPath)
		if err != nil {
			return count, unmatched, first, err
		}
		targetDocuments, targetAliases, err = activeClosureDocuments(context.Background(), target)
		if err != nil {
			return count, unmatched, first, err
		}
	}
	var rebound []closureledger.Document
	var retirements []artifact.AliasBinding
	retired := map[string]bool{}
	index := closurescan.CompileRebindIndex(candidates)
	for _, document := range targetDocuments {
		if len(document.Bindings) != 1 {
			continue
		}
		binding := document.Bindings[0]
		alias, err := closureledger.ActiveAlias(binding)
		if err != nil {
			return count, unmatched, first, err
		}
		current, matched, _, err := rebindClosure(index, document, reviewCallsites)
		if err != nil {
			return count, unmatched, first, err
		}
		if targetAliases[alias] != document.ID || retired[alias] {
			continue
		}
		moved := false
		if matched {
			currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
			if err != nil {
				return count, unmatched, first, err
			}
			moved = currentAlias != alias
		}
		if !matched || moved {
			retirements = append(retirements, artifact.AliasBinding{Name: alias, Target: document.ID, Previous: &document.ID, Remove: true})
			retired[alias] = true
		}
	}
	for _, document := range documents {
		if len(document.Bindings) != 1 {
			unmatched++
			first = cmp.Or(first, document.Name+":bindings")
			continue
		}
		previousAlias, err := closureledger.ActiveAlias(document.Bindings[0])
		activeSource := sourceAliases[previousAlias] == document.ID
		if err != nil || !activeSource && !recoverRetired {
			unmatched++
			first = cmp.Or(first, document.Name+":source-alias")
			continue
		}
		current, matched, reason, err := rebindClosure(index, document, reviewCallsites)
		if err != nil {
			return count, unmatched, first, err
		}
		if !matched {
			unmatched++
			first = cmp.Or(first, document.Name+":"+reason)
			continue
		}
		if recoverRetired && !activeSource {
			currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
			if err != nil {
				return count, unmatched, first, err
			}
			if targetAliases[currentAlias] == current.ID || slices.ContainsFunc(rebound, func(existing closureledger.Document) bool {
				alias, aliasErr := closureledger.ActiveAlias(existing.Bindings[0])
				return aliasErr == nil && alias == currentAlias
			}) {
				continue
			}
		}
		if sameStore && activeSource && current.ID == document.ID {
			continue
		}
		currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
		if err != nil {
			return count, unmatched, first, err
		}
		if sameStore && currentAlias != previousAlias && !retired[previousAlias] {
			retirements = append(retirements, artifact.AliasBinding{Name: previousAlias, Target: document.ID, Previous: &document.ID, Remove: true})
			retired[previousAlias] = true
		}
		rebound = append(rebound, current)
	}
	if recoverRetired {
		exact := map[string]closureledger.Document{}
		for _, document := range documents {
			if len(document.Bindings) != 1 {
				continue
			}
			key := exactClosureKey(document.Bindings[0], document.Value)
			if _, found := exact[key]; !found {
				exact[key] = document
			}
		}
		for _, candidate := range candidates {
			binding, err := candidate.Binding()
			if err != nil {
				return count, unmatched, first, err
			}
			document, found := exact[exactClosureKey(binding, candidate.ValueJSON())]
			if !found {
				continue
			}
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return count, unmatched, first, err
			}
			if targetAliases[alias] == document.ID && !retired[alias] {
				continue
			}
			if index := slices.IndexFunc(rebound, func(existing closureledger.Document) bool {
				currentAlias, aliasErr := closureledger.ActiveAlias(existing.Bindings[0])
				return aliasErr == nil && currentAlias == alias
			}); index >= 0 {
				rebound[index] = document
			} else {
				rebound = append(rebound, document)
			}
		}
		historicalGroups := map[string][]closureledger.Document{}
		groupNames := map[string]map[string]bool{}
		for _, document := range documents {
			if len(document.Bindings) != 1 {
				continue
			}
			key := recoveryGroupKey(document.Bindings[0], document.Value)
			if groupNames[key] == nil {
				groupNames[key] = map[string]bool{}
			}
			if groupNames[key][document.Name] {
				continue
			}
			groupNames[key][document.Name] = true
			historicalGroups[key] = append(historicalGroups[key], document)
		}
		currentGroups := map[string][]closurescan.Candidate{}
		for _, candidate := range candidates {
			if candidate.Kind != closureledger.BindingConstant && !candidate.Policy {
				continue
			}
			binding, err := candidate.Binding()
			if err != nil {
				return count, unmatched, first, err
			}
			key := recoveryGroupKey(binding, candidate.ValueJSON())
			currentGroups[key] = append(currentGroups[key], candidate)
		}
		groupKeys := slices.Sorted(maps.Keys(currentGroups))
		for _, key := range groupKeys {
			current := currentGroups[key]
			historical := historicalGroups[key]
			slices.SortFunc(current, func(left, right closurescan.Candidate) int {
				return cmp.Or(cmp.Compare(left.Line, right.Line), cmp.Compare(left.Name, right.Name))
			})
			slices.SortFunc(historical, func(left, right closureledger.Document) int {
				return cmp.Or(cmp.Compare(left.Bindings[0].Line, right.Bindings[0].Line), cmp.Compare(left.Name, right.Name))
			})
			usedHistorical := make([]bool, len(historical))
			unresolved := make([]closurescan.Candidate, 0, len(current))
			for _, candidate := range current {
				binding, err := candidate.Binding()
				if err != nil {
					return count, unmatched, first, err
				}
				if document, found := exact[exactClosureKey(binding, candidate.ValueJSON())]; found {
					if historicalIndex := slices.IndexFunc(historical, func(existing closureledger.Document) bool {
						return existing.ID == document.ID
					}); historicalIndex >= 0 {
						usedHistorical[historicalIndex] = true
					}
					continue
				}
				unresolved = append(unresolved, candidate)
			}
			recoverCandidate := func(candidate closurescan.Candidate, previous closureledger.Document) error {
				binding, err := candidate.Binding()
				if err != nil {
					return err
				}
				fixture := previous.Fixture
				if fixture == previous.Bindings[0].Owner {
					fixture = binding.Owner
				}
				currentDocument, err := closureledger.New(
					previous.Name, previous.Value, previous.Tier, previous.Status, previous.Understanding,
					[]closureledger.SourceBinding{binding}, previous.ClosurePath, previous.RerankTrigger, fixture,
				)
				if err != nil {
					return err
				}
				alias, err := closureledger.ActiveAlias(binding)
				if err != nil {
					return err
				}
				if targetAliases[alias] == currentDocument.ID && !retired[alias] {
					return nil
				}
				if reboundIndex := slices.IndexFunc(rebound, func(existing closureledger.Document) bool {
					currentAlias, aliasErr := closureledger.ActiveAlias(existing.Bindings[0])
					return aliasErr == nil && currentAlias == alias
				}); reboundIndex >= 0 {
					rebound[reboundIndex] = currentDocument
				} else {
					rebound = append(rebound, currentDocument)
				}
				return nil
			}
			// A syntax rewrite can remove some members of an otherwise stable
			// literal group. Preserve exact same-line decisions first; unlike
			// ordinal pairing, this remains unambiguous when group counts shrink.
			historicalByLine := map[int]closureledger.Document{}
			for index, previous := range historical {
				if usedHistorical[index] {
					continue
				}
				line := previous.Bindings[0].Line
				if _, found := historicalByLine[line]; !found {
					historicalByLine[line] = previous
				}
			}
			currentByLine := map[int]closurescan.Candidate{}
			ambiguousCurrentLine := map[int]bool{}
			for _, candidate := range unresolved {
				if _, found := currentByLine[candidate.Line]; found {
					ambiguousCurrentLine[candidate.Line] = true
				} else {
					currentByLine[candidate.Line] = candidate
				}
			}
			remaining := unresolved[:0]
			for _, candidate := range unresolved {
				previous, historicalFound := historicalByLine[candidate.Line]
				if historicalFound && !ambiguousCurrentLine[candidate.Line] {
					if err := recoverCandidate(candidate, previous); err != nil {
						return count, unmatched, first, err
					}
					for index, document := range historical {
						if document.ID == previous.ID {
							usedHistorical[index] = true
							break
						}
					}
					continue
				}
				remaining = append(remaining, candidate)
			}
			closestHistorical := func(candidate closurescan.Candidate) (closureledger.Document, bool) {
				var closest closureledger.Document
				bestDelta, found, ambiguous := 0, false, false
				for index, previous := range historical {
					if usedHistorical[index] {
						continue
					}
					difference := sourceOrderDifference(candidateSourceOrder(candidate), bindingSourceOrder(previous.Bindings[0]))
					switch {
					case !found || difference < bestDelta:
						closest, bestDelta, found, ambiguous = previous, difference, true, false
					case difference == bestDelta:
						ambiguous = true
					}
				}
				return closest, found && !ambiguous
			}
			closestCurrent := func(previous closureledger.Document, candidates []closurescan.Candidate) (closurescan.Candidate, bool) {
				var closest closurescan.Candidate
				bestDelta, found, ambiguous := 0, false, false
				for _, candidate := range candidates {
					difference := sourceOrderDifference(candidateSourceOrder(candidate), bindingSourceOrder(previous.Bindings[0]))
					switch {
					case !found || difference < bestDelta:
						closest, bestDelta, found, ambiguous = candidate, difference, true, false
					case difference == bestDelta:
						ambiguous = true
					}
				}
				return closest, found && !ambiguous
			}
			// When removed syntax shrinks a group and prior edits shift lines,
			// recover only mutual unique nearest neighbours. This is deterministic
			// and refuses rather than guessing whenever either side is ambiguous.
			for progress := true; progress; {
				progress = false
				var next []closurescan.Candidate
				for _, candidate := range remaining {
					previous, found := closestHistorical(candidate)
					if !found {
						next = append(next, candidate)
						continue
					}
					nearest, mutual := closestCurrent(previous, remaining)
					if !mutual || nearest.StructuralID != candidate.StructuralID {
						next = append(next, candidate)
						continue
					}
					if err := recoverCandidate(candidate, previous); err != nil {
						return count, unmatched, first, err
					}
					for index, document := range historical {
						if document.ID == previous.ID {
							usedHistorical[index] = true
							break
						}
					}
					progress = true
				}
				remaining = next
			}
			available := make([]closureledger.Document, 0, len(historical))
			for index, previous := range historical {
				if !usedHistorical[index] {
					available = append(available, previous)
				}
			}
			if len(remaining) != len(available) {
				continue
			}
			for index, candidate := range remaining {
				if err := recoverCandidate(candidate, available[index]); err != nil {
					return count, unmatched, first, err
				}
			}
		}
	}
	fixtures, err := closureFixtureImports(context.Background(), source, target, rebound)
	sameView := target == source
	closeErr := source.Close()
	if !sameView {
		closeErr = errors.Join(closeErr, target.Close())
	}
	source, target = nil, nil
	if err != nil || closeErr != nil {
		return count, unmatched, first, errors.Join(err, closeErr)
	}
	if rebound == nil && retirements == nil {
		return count, unmatched, first, nil
	}
	if _, _, err := commitClosureDocuments(root, destinationPath, rebound, retirements, fixtures); err != nil {
		return count, unmatched, first, err
	}
	return len(rebound), unmatched, first, nil
}

func exactClosureKey(binding closureledger.SourceBinding, value json.RawMessage) string {
	return strings.Join([]string{
		string(binding.Kind), binding.Package, binding.File, binding.Scope, binding.Name,
		binding.StructuralID, binding.Expression, binding.SourceID, binding.CallsiteID, binding.Owner.String(), string(value),
	}, "\x00")
}

func recoveryGroupKey(binding closureledger.SourceBinding, value json.RawMessage) string {
	name := ""
	if binding.Kind == closureledger.BindingConstant {
		name = binding.Name
	}
	return strings.Join([]string{
		string(binding.Kind), binding.Package, binding.File, binding.Scope, name, binding.Expression, string(value),
	}, "\x00")
}

func sourceOrderDifference(left, right int) int {
	if left > right {
		return left - right
	}
	return right - left
}

func candidateSourceOrder(candidate closurescan.Candidate) int {
	return candidate.Line
}

func bindingSourceOrder(binding closureledger.SourceBinding) int {
	return binding.Line
}

func retiredClosureDocuments(ctx context.Context, store *overgodb.Store, active []closureledger.Document) ([]closureledger.Document, error) {
	activeIDs := map[artifact.ID]bool{}
	for _, document := range active {
		activeIDs[document.ID] = true
	}
	var documents []closureledger.Document
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: closureledger.MediaType, Schema: closureledger.Schema,
		}}, Order: overgodb.DocumentNewestFirst,
	}, closureledger.Parse, func(_ overgodb.DocumentView, document closureledger.Document) error {
		if activeIDs[document.ID] {
			return nil
		}
		documents = append(documents, document)
		return nil
	})
	return documents, err
}

func rebindClosure(
	index closurescan.RebindIndex,
	document closureledger.Document,
	reviewCallsites bool,
) (closureledger.Document, bool, string, error) {
	if reviewCallsites {
		return index.RebindReviewed(document)
	}
	return index.Rebind(document)
}

func closureFixtureImports(ctx context.Context, source, target *overgodb.Store, documents []closureledger.Document) ([]artifact.Descriptor, error) {
	var descriptors []artifact.Descriptor
	for _, document := range documents {
		if _, found, err := target.Artifact(ctx, document.Fixture); err != nil {
			return nil, err
		} else if found || slices.ContainsFunc(document.Bindings, func(binding closureledger.SourceBinding) bool { return binding.Owner == document.Fixture }) {
			continue
		}
		descriptor, found, err := source.Artifact(ctx, document.Fixture)
		if err != nil || !found {
			return nil, errors.Join(err, fmt.Errorf("closure fixture is absent: %s", document.Fixture))
		}
		descriptors = append(descriptors, descriptor)
	}
	return descriptors, nil
}

func activeClosureDocuments(ctx context.Context, store *overgodb.Store) ([]closureledger.Document, map[string]artifact.ID, error) {
	documents := make([]closureledger.Document, 0)
	aliases := map[string]artifact.ID{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: closureledger.MediaType, Schema: closureledger.Schema,
		}}, AliasPrefixes: []string{closureledger.ActiveAliasPrefix}, Order: overgodb.DocumentOldestFirst,
	}, closureledger.Parse, func(view overgodb.DocumentView, document closureledger.Document) error {
		documents = append(documents, document)
		for _, name := range view.Aliases {
			aliases[name] = document.ID
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return documents, aliases, nil
}

func writeCensusText(destination io.Writer, census closurescan.Census) error {
	counts := census.Counts
	_, err := fmt.Fprintf(destination,
		"closure-scan census %s\nsource %s\nfiles production=%d test=%d\n"+
			"surfaces named=%d inline=%d assumptions=%d test_policy=%d\n"+
			"literal_classes structural=%d mathematical=%d format=%d capacity=%d policy=%d model_fact=%d unknown=%d\n"+
			"tests total=%d fixture=%d assertion=%d policy_copy=%d\n"+
			"repeated groups=%d sites=%d\ndetail: rerun with -format json\n",
		census.Schema, census.Source, counts.ProductionFiles, counts.TestFiles,
		counts.NamedConstants, counts.InlineLiterals, counts.AssumptionHints, counts.TestPolicyCopies,
		counts.Structural, counts.Mathematical, counts.Format, counts.Capacity, counts.Policy, counts.ModelFact, counts.Unknown,
		counts.TestLiterals, counts.TestFixtures, counts.TestAssertions, counts.TestPolicyCopies,
		counts.RepeatedGroups, counts.RepeatedSites)
	return err
}

func reportLiterals(sites []closurescan.LiteralSite, limit int) {
	fmt.Printf("closure-scan: %d classified production numeric literals (named constants excluded)\n", len(sites))
	fmt.Printf("%-14s %-18s %-16s %-24s %s\n", "class", "context", "value", "scope", "source")
	for _, site := range sites[:min(len(sites), limit)] {
		fmt.Printf("%-14s %-18s %-16s %-24s %s:%d\n", site.Class, site.Context, site.Value, site.Scope, site.File, site.Line)
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
	for _, site := range sites[:min(len(sites), limit)] {
		fmt.Printf("%-14s %-18s %-16s %-24s %s:%d\n",
			site.Class, site.Context, site.Value, site.Scope, site.File, site.Line)
	}
}

func reportAssumptions(hints []closurescan.AssumptionHint, limit int) {
	fmt.Printf("closure-scan: %d syntax-derived assumption hints\n", len(hints))
	fmt.Printf("%-32s %-7s %-24s %s\n", "binding", "policy", "scope", "source")
	for _, hint := range hints[:min(len(hints), limit)] {
		fmt.Printf("%-32s %-7t %-24s %s:%d\n",
			hint.Candidate().Name, hint.Policy, hint.Scope, hint.File, hint.Line)
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
	for _, row := range candidates[:min(len(candidates), limit)] {
		fmt.Printf("%-6d %-44s %-16s %-24s %s:%d\n", row.Score, row.Name, row.Value, row.Scope, row.File, row.Line)
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
		byKey[row.LegacyDeclarationKey()] = row
	}
	documents := make([]closureledger.Document, 0, len(triage.Rows))
	for _, row := range triage.Rows {
		found, ok := byKey[(closurescan.Candidate{
			Kind: row.Kind, File: row.File, Scope: row.Scope, Line: row.Line, Name: row.Name,
		}).DeclarationKey()]
		if !ok {
			return fmt.Errorf("triage row %s not found by scan in %s (stale triage?)", row.Name, row.File)
		}
		valueJSON := found.ValueJSON()
		binding, err := found.Binding()
		if err != nil {
			return err
		}
		document, err := closureledger.New(
			row.Name, valueJSON,
			closureledger.Tier(row.Tier), closureledger.Status(row.Status),
			row.Understanding, []closureledger.SourceBinding{binding},
			row.ClosurePath, row.RerankTrigger, binding.Owner,
		)
		if err != nil {
			return fmt.Errorf("row %s: %w", row.Name, err)
		}
		documents = append(documents, document)
	}
	owners, commit, err := commitClosureDocuments(root, filepath.Join(root, storePath), documents, nil, nil)
	if err != nil {
		return err
	}
	fmt.Printf("emitted %d closure documents (%d owner files) commit %x\n",
		len(documents), owners, commit[:8])
	return nil
}

func commitClosureDocuments(root, storePath string, documents []closureledger.Document, retirements []artifact.AliasBinding, fixtures []artifact.Descriptor) (int, artifact.CommitID, error) {
	store, err := overgodb.Open(storePath)
	if err != nil {
		return 0, artifact.CommitID{}, err
	}
	defer store.Close()
	batch := artifact.Batch{Aliases: retirements}
	batchedContents := map[artifact.ID]bool{}
	for _, fixture := range fixtures {
		if _, found, err := store.Artifact(context.Background(), fixture.ID); err != nil {
			return 0, artifact.CommitID{}, err
		} else if !found {
			batch.Artifacts = append(batch.Artifacts, fixture)
		}
	}
	files := map[string]bool{}
	for _, document := range documents {
		for _, binding := range document.Bindings {
			if !files[binding.File] {
				id, descriptor, location, err := fileArtifact(root, binding.File)
				if err != nil || id != binding.Owner {
					return 0, artifact.CommitID{}, fmt.Errorf("closure %s: inconsistent source owner", binding.Name)
				}
				files[binding.File] = true
				if _, found, err := store.Artifact(context.Background(), descriptor.ID); err != nil {
					return 0, artifact.CommitID{}, err
				} else if !found {
					batch.Artifacts = append(batch.Artifacts, descriptor)
				}
				locations, err := store.Locations(context.Background(), descriptor.ID)
				if err != nil {
					return 0, artifact.CommitID{}, err
				}
				if !slices.Contains(locations, location.Location) {
					batch.Locations = append(batch.Locations, location)
				}
			}
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return 0, artifact.CommitID{}, err
			}
			active := artifact.AliasBinding{Name: alias, Target: document.ID}
			if previous, found, err := artifact.ResolveAlias(context.Background(), store, alias); err != nil {
				return 0, artifact.CommitID{}, err
			} else if found {
				if previous == document.ID {
					continue
				}
				active.Previous = &previous
			}
			// A compacted target flattens supersede history, so one alias can
			// collide within a single import: several source revisions of one
			// decision rebind the same name, or a rebind lands on an alias the
			// retirement pass already scheduled. One batch admits one binding
			// per name; a live rebind replaces a scheduled retirement outright,
			// and among rebinds the newest source document wins because
			// documents iterate in commit order.
			if index := slices.IndexFunc(batch.Aliases, func(bound artifact.AliasBinding) bool {
				return bound.Name == alias
			}); index >= 0 {
				batch.Aliases[index] = active
				continue
			}
			batch.Aliases = append(batch.Aliases, active)
		}
		content, err := document.Content()
		if err != nil {
			return 0, artifact.CommitID{}, err
		}
		if found, err := store.HasContent(context.Background(), document.ID); err != nil {
			return 0, artifact.CommitID{}, err
		} else if !found && !batchedContents[document.ID] {
			batch.Contents = append(batch.Contents, content)
			batch.Lineage = append(batch.Lineage, document.Lineage()...)
			batchedContents[document.ID] = true
		}
	}
	aliases := batch.Aliases[:0]
	for _, binding := range batch.Aliases {
		current, found, err := artifact.ResolveAlias(context.Background(), store, binding.Name)
		if err != nil {
			return 0, artifact.CommitID{}, err
		}
		if binding.Remove {
			if !found {
				continue
			}
			binding.Target = current
			binding.Previous = &current
		} else if found {
			if current == binding.Target {
				continue
			}
			binding.Previous = &current
		} else {
			binding.Previous = nil
		}
		aliases = append(aliases, binding)
	}
	batch.Aliases = aliases
	if batch.Empty() {
		head, _ := store.Head()
		return len(files), head, nil
	}
	encoded, err := json.Marshal(batch)
	if err != nil {
		return 0, artifact.CommitID{}, err
	}
	head, _ := store.Head()
	digestInput := append([]byte(head.String()+"\n"), encoded...)
	digest := sha256.Sum256(digestInput)
	batch.Key = "closure-scan/" + hex.EncodeToString(digest[:])
	commit, err := store.Commit(context.Background(), batch)
	if err != nil {
		return 0, artifact.CommitID{}, err
	}
	return len(files), commit, nil
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
