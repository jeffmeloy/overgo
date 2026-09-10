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

type closureAliasReview struct {
	Reviewed, Retired, Preserved int
}

type closureCommitOperation string

const (
	closurePublishOperation             closureCommitOperation = "closure-scan/publish/"
	closureImportOperation              closureCommitOperation = "closure-scan/import/"
	closureRebindOperation              closureCommitOperation = "closure-scan/rebind/"
	closureRetireUnmatchedOperation     closureCommitOperation = "closure-scan/retire-unmatched/"
	closureRemediateRecoveryOperation   closureCommitOperation = "closure-scan/remediate-reactivation/"
	closureRestoreReviewedHeadOperation closureCommitOperation = "closure-scan/restore-reviewed-head/"
)

func main() {
	triagePath := flag.String("triage", "", "triage JSON ({rows:[{kind,name,file,scope,line,tier,status,understanding,closure_path,rerank_trigger}]})")
	storePath := flag.String("store", "overgodb-store", "OvergoDB store directory (emit mode)")
	limit := flag.Int("limit", 40, "report mode: top-N candidates to print")
	literals := flag.Bool("literals", false, "report classified production literals instead of declared constants")
	testLiterals := flag.Bool("test-literals", false, "report classified test literals and production overlaps")
	assumptions := flag.Bool("assumptions", false, "report syntax-derived distribution, geometry, and shape hints")
	census := flag.Bool("census", false, "report complete source denominators and consolidation pressure")
	inventoryUnclassified := flag.Bool("inventory-unclassified", false, "report every current constant without exact active authority and matching history")
	inventoryUnclassifiedPolicy := flag.Bool("inventory-unclassified-policy", false, "report every current production policy candidate without exact active authority and matching history")
	propose := flag.String("propose", "", "emit ready-to-apply triage rows with exact scanner coordinates for the named uncatalogued candidates (comma-separated)")
	format := flag.String("format", "text", "structured report format: text or json")
	publish := flag.Bool("publish", false, "publish census evidence to OvergoDB")
	importStore := flag.String("import-store", "", "import matching active decisions; same-store mode safely rebinds unambiguous history")
	reviewCallsites := flag.Bool("review-callsites", false, "with -import-store: accept reviewed callsite drift")
	retireUnmatched := flag.Bool("retire-unmatched", false, "with same-store -import-store: CAS-retire active decisions that cannot rebind")
	remediateCommit := flag.String("remediate-reactivation-commit", "", "reviewed reactivation commit to remediate")
	var remediateSequence uint64
	flag.Uint64Var(&remediateSequence, "remediate-reactivation-sequence", remediateSequence, "reviewed reactivation sequence to remediate")
	remediateHead := flag.String("remediate-expected-head", "", "required reviewed current store head for reactivation remediation")
	var remediateHeadSequence uint64
	flag.Uint64Var(&remediateHeadSequence, "remediate-expected-sequence", remediateHeadSequence, "required reviewed current store sequence for reactivation remediation")
	var remediateExpectedCount int
	flag.IntVar(&remediateExpectedCount, "remediate-expected-count", remediateExpectedCount, "required reviewed active-alias count for reactivation remediation")
	confirmUnverifiedRecovery := flag.Bool("confirm-unverified-recovery", false, "attest that a selected unverified reactivation was a reviewed recovery operation")
	restoreSourceHead := flag.String("restore-reviewed-head", "", "reviewed historical store head whose closure alias facet should be restored")
	var restoreSourceSequence uint64
	flag.Uint64Var(&restoreSourceSequence, "restore-reviewed-sequence", restoreSourceSequence, "reviewed historical sequence whose closure alias facet should be restored")
	restoreExpectedHead := flag.String("restore-current-head", "", "required exact current store head for reviewed alias restoration")
	var restoreExpectedSequence uint64
	flag.Uint64Var(&restoreExpectedSequence, "restore-current-sequence", restoreExpectedSequence, "required exact current store sequence for reviewed alias restoration")
	restoreExpectedAliases := flag.Int("restore-expected-aliases", unreviewedRestoreCount, "confirmed reviewed alias count; omit for dry-run")
	restoreExpectedChanges := flag.Int("restore-expected-changes", unreviewedRestoreCount, "confirmed reviewed alias delta count; omit for dry-run")
	restoreExpectedStale := flag.Int("restore-expected-stale", unreviewedRestoreCount, "confirmed predicted stale binding count; omit for dry-run")
	restoreExpectedAuthorityDigest := flag.String("restore-expected-authority-digest", "", "confirmed reviewed permanent-authority result digest; omit for dry-run")
	confirmAliasRestore := flag.Bool("confirm-reviewed-restore", false, "commit the exact reviewed closure alias restoration")
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
	if (*reviewCallsites || *retireUnmatched) && *importStore == "" {
		fatal(errors.New("-review-callsites and -retire-unmatched require -import-store"))
	}
	remediationMode := *remediateCommit != "" || remediateSequence != 0 || *remediateHead != "" ||
		remediateHeadSequence != 0 || remediateExpectedCount != 0 || *confirmUnverifiedRecovery
	if remediationMode && (*remediateCommit == "" || remediateSequence == 0 || *remediateHead == "" || remediateHeadSequence == 0) {
		fatal(errors.New("reactivation remediation requires commit, sequence, expected head, expected sequence, and expected count"))
	}
	if remediationMode && remediateExpectedCount <= 0 {
		fatal(errors.New("reactivation remediation expected count must be positive"))
	}
	restoreMode := *restoreSourceHead != "" || restoreSourceSequence != 0 || *restoreExpectedHead != "" ||
		restoreExpectedSequence != 0 || *confirmAliasRestore
	if restoreMode && (*restoreSourceHead == "" || restoreSourceSequence == 0 ||
		*restoreExpectedHead == "" || restoreExpectedSequence == 0) {
		fatal(errors.New("reviewed alias restore requires source and current commit/sequence coordinates"))
	}
	if *confirmAliasRestore && !validClosureAuthorityDigest(*restoreExpectedAuthorityDigest) {
		fatal(errors.New("confirmed alias restore requires -restore-expected-authority-digest as exact lowercase SHA-256 hex"))
	}
	if !*confirmAliasRestore && *restoreExpectedAuthorityDigest != "" {
		fatal(errors.New("-restore-expected-authority-digest requires -confirm-reviewed-restore"))
	}
	root, err := os.Getwd()
	if err != nil {
		fatal(err)
	}
	modes := 0
	for _, enabled := range []bool{
		*literals, *testLiterals, *assumptions, *census, *inventoryUnclassified, *inventoryUnclassifiedPolicy, *importStore != "",
		remediationMode, restoreMode, *checkScope != "", *checkAll, *checkTests,
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
	if *propose != "" {
		snapshot := mustSnapshot(root)
		store, err := overgodb.OpenReadOnly(filepath.Join(root, *storePath))
		if err != nil {
			fatal(err)
		}
		proposal, proposeErr := proposeTriageRows(
			context.Background(), snapshot, store, strings.Split(*propose, ","),
		)
		closeErr := store.Close()
		if proposeErr != nil || closeErr != nil {
			fatal(errors.Join(proposeErr, closeErr))
		}
		if err := clioptions.WritePrettyJSON(os.Stdout, proposal); err != nil {
			fatal(err)
		}
		return
	}
	if *inventoryUnclassified || *inventoryUnclassifiedPolicy {
		snapshot := mustSnapshot(root)
		store, err := overgodb.OpenReadOnly(filepath.Join(root, *storePath))
		if err != nil {
			fatal(err)
		}
		var report unclassifiedReport
		var reportErr error
		if *inventoryUnclassifiedPolicy {
			report, reportErr = buildUnclassifiedPolicyReport(context.Background(), snapshot, store)
		} else {
			report, reportErr = buildUnclassifiedReport(context.Background(), snapshot, store)
		}
		closeErr := store.Close()
		if reportErr != nil || closeErr != nil {
			fatal(errors.Join(reportErr, closeErr))
		}
		switch *format {
		case "text":
			if *inventoryUnclassifiedPolicy {
				err = writeUnclassifiedPolicyReport(os.Stdout, report)
			} else {
				err = writeUnclassifiedReport(os.Stdout, report)
			}
		case "json":
			err = clioptions.WritePrettyJSON(os.Stdout, report)
		default:
			err = fmt.Errorf("unknown inventory format %q", *format)
		}
		if err != nil {
			fatal(err)
		}
		return
	}
	if remediationMode {
		selector, err := parseClosureRemediationSelector(
			*remediateCommit, remediateSequence, *remediateHead, remediateHeadSequence,
			remediateExpectedCount, *confirmUnverifiedRecovery,
		)
		if err != nil {
			fatal(err)
		}
		result, err := remediateClosureReactivations(context.Background(), filepath.Join(root, *storePath), selector)
		if err != nil {
			fatal(err)
		}
		if err := clioptions.WritePrettyJSON(os.Stdout, result); err != nil {
			fatal(err)
		}
		return
	}
	if restoreMode {
		sourceHead, err := parseClosureCommitID("reviewed restore head", *restoreSourceHead)
		if err != nil {
			fatal(err)
		}
		expectedHead, err := parseClosureCommitID("current restore head", *restoreExpectedHead)
		if err != nil {
			fatal(err)
		}
		result, err := restoreClosureAliasesAtReviewedHead(
			context.Background(), filepath.Join(root, *storePath), mustSnapshot(root),
			closureAliasRestoreSelector{
				SourceHead: sourceHead, SourceSequence: restoreSourceSequence,
				ExpectedHead: expectedHead, ExpectedSequence: restoreExpectedSequence,
				ExpectedAliases: *restoreExpectedAliases, ExpectedChanges: *restoreExpectedChanges,
				ExpectedStale: *restoreExpectedStale, ExpectedAuthorityDigest: *restoreExpectedAuthorityDigest,
				Confirm: *confirmAliasRestore,
			},
		)
		if err != nil {
			fatal(err)
		}
		if err := clioptions.WritePrettyJSON(os.Stdout, result); err != nil {
			fatal(err)
		}
		return
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
			store, err := overgodb.OpenContext(context.Background(), filepath.Join(root, *storePath))
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
		count, unmatched, first, aliases, err := importClosureDocuments(
			root, *storePath, *importStore, mustSnapshot(root), *reviewCallsites, *retireUnmatched,
		)
		if err != nil {
			fatal(err)
		}
		fmt.Printf(
			"imported %d closure document(s), unmatched=%d first=%s aliases_reviewed=%d retired=%d preserved=%d\n",
			count, unmatched, first, aliases.Reviewed, aliases.Retired, aliases.Preserved,
		)
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
	documents, aliases, err := activeClosureDocuments(context.Background(), store)
	if err != nil {
		return err
	}
	if requirements.noStale {
		issues, err := closurescan.ValidateActiveBindings(snapshot, scopedDocuments(documents, prefixes), aliases)
		if err != nil {
			return err
		}
		if len(issues) > 0 {
			// Every stale binding prints: remediation is a batch (one
			// edit shifts every literal offset after it), and a report
			// naming only the first costs one gate run per binding.
			for _, issue := range issues {
				fmt.Printf("stale %s %s:%s\n", issue.Kind, issue.File, issue.Name)
			}
			return fmt.Errorf("%d scoped closure binding(s) stale; first=%s:%s", len(issues), issues[0].File, issues[0].Name)
		}
	}
	active, err := compileActiveClosures(documents, aliases)
	if err != nil {
		return err
	}
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

func compileActiveClosures(
	documents []closureledger.Document,
	aliases map[string]artifact.ID,
) (map[string]closureledger.Document, error) {
	active := make(map[string]closureledger.Document, len(documents))
	for _, document := range documents {
		for _, binding := range document.Bindings {
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return nil, err
			}
			if aliases[alias] != document.ID {
				continue
			}
			key := (closurescan.Candidate{
				Kind: binding.Kind, Package: binding.Package, File: binding.File, Scope: binding.Scope,
				Line: binding.Line, Name: binding.Name, StructuralID: binding.StructuralID,
			}).DeclarationKey()
			active[key] = document
		}
	}
	return active, nil
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
	active, aliases, err := activeClosureDocuments(ctx, store)
	if err != nil {
		return closurescan.CensusEvidence{}, err
	}
	issues, err := closurescan.ValidateActiveBindings(snapshot, active, aliases)
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
	reviewCallsites, retireUnmatched bool,
) (count, unmatched int, first string, aliases closureAliasReview, finalErr error) {
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateAll)
	if err != nil {
		return count, unmatched, first, aliases, err
	}
	destinationPath := filepath.Join(root, storePath)
	sourceInfo, sourceErr := os.Stat(sourcePath)
	destinationInfo, destinationErr := os.Stat(destinationPath)
	sameStore := sourceErr == nil && destinationErr == nil && os.SameFile(sourceInfo, destinationInfo)
	if retireUnmatched && !sameStore {
		return count, unmatched, first, aliases, errors.New("-retire-unmatched requires source and destination to be the same store")
	}
	source, err := overgodb.OpenReadOnly(sourcePath)
	if err != nil {
		return count, unmatched, first, aliases, err
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
		return count, unmatched, first, aliases, err
	}
	if sameStore {
		aliases.Reviewed = len(sourceAliases)
		aliases.Preserved = len(sourceAliases)
	}
	historicalDocuments := documents
	recoveryAuthority := map[string]artifact.ID{}
	if sameStore && !retireUnmatched {
		historicalDocuments, err = allClosureDocuments(context.Background(), source)
		if err != nil {
			return count, unmatched, first, aliases, err
		}
		recoveryAuthority, err = closureRecoveryAuthority(context.Background(), source, historicalDocuments)
		if err != nil {
			return count, unmatched, first, aliases, err
		}
	}
	targetDocuments, targetAliases := documents, sourceAliases
	if sameStore {
		target = source
	} else {
		target, err = overgodb.OpenContext(context.Background(), destinationPath)
		if err != nil {
			return count, unmatched, first, aliases, err
		}
		targetDocuments, targetAliases, err = activeClosureDocuments(context.Background(), target)
		if err != nil {
			return count, unmatched, first, aliases, err
		}
	}
	reviewedTargetHead, _ := target.Head()
	var rebound []closureledger.Document
	var retirements []artifact.AliasBinding
	var unmatchedRetirements []artifact.AliasBinding
	retired := map[string]bool{}
	index := closurescan.CompileRebindIndex(candidates)
	if !sameStore {
		for _, document := range targetDocuments {
			if len(document.Bindings) != 1 {
				continue
			}
			binding := document.Bindings[0]
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return count, unmatched, first, aliases, err
			}
			current, matched, _, err := rebindClosure(index, document, reviewCallsites)
			if err != nil {
				return count, unmatched, first, aliases, err
			}
			if targetAliases[alias] != document.ID || retired[alias] {
				continue
			}
			moved := false
			if matched {
				currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
				if err != nil {
					return count, unmatched, first, aliases, err
				}
				moved = currentAlias != alias
			}
			if !matched || moved {
				retirements = append(retirements, artifact.AliasBinding{Name: alias, Target: document.ID, Previous: &document.ID, Remove: true})
				retired[alias] = true
			}
		}
	}
	claimedAliases := make(map[string]bool, len(sourceAliases))
	for alias := range sourceAliases {
		claimedAliases[alias] = true
	}
	type contentRetry struct {
		document      closureledger.Document
		previousAlias string
		reason        string
	}
	var contentRetries []contentRetry
	for _, document := range documents {
		if len(document.Bindings) != 1 {
			unmatched++
			first = cmp.Or(first, document.Name+":bindings")
			continue
		}
		previousAlias, err := closureledger.ActiveAlias(document.Bindings[0])
		if err != nil || sourceAliases[previousAlias] != document.ID {
			unmatched++
			first = cmp.Or(first, document.Name+":source-alias")
			continue
		}
		current, matched, reason, err := rebindClosure(index, document, reviewCallsites)
		if err != nil {
			return count, unmatched, first, aliases, err
		}
		if !matched {
			contentRetries = append(contentRetries, contentRetry{
				document: document, previousAlias: previousAlias, reason: reason,
			})
			continue
		}
		if sameStore && current.ID == document.ID {
			continue
		}
		currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
		if err != nil {
			return count, unmatched, first, aliases, err
		}
		claimedAliases[currentAlias] = true
		if sameStore && currentAlias != previousAlias && !retired[previousAlias] {
			retirements = append(retirements, artifact.AliasBinding{Name: previousAlias, Target: document.ID, Previous: &document.ID, Remove: true})
			retired[previousAlias] = true
		}
		rebound = append(rebound, current)
	}
	// Content-matched recovery runs after every structural rebind has
	// claimed its alias, so the only free candidates left are genuinely
	// new sites; an offset-shifted successor with the same file, scope,
	// and exact value inherits the reviewed closure instead of being
	// retired and retyped.
	for _, retry := range contentRetries {
		document, previousAlias, reason := retry.document, retry.previousAlias, retry.reason
		current, matched, _, err := closurescan.ContentMatchedRebind(
			document, candidates, func(alias string) bool { return claimedAliases[alias] },
		)
		if err != nil {
			return count, unmatched, first, aliases, err
		}
		if matched {
			currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
			if err != nil {
				return count, unmatched, first, aliases, err
			}
			claimedAliases[currentAlias] = true
			if sameStore && currentAlias != previousAlias && !retired[previousAlias] {
				retirements = append(retirements, artifact.AliasBinding{Name: previousAlias, Target: document.ID, Previous: &document.ID, Remove: true})
				retired[previousAlias] = true
			}
			rebound = append(rebound, current)
			continue
		}
		unmatched++
		first = cmp.Or(first, document.Name+":"+reason)
		if sameStore && retireUnmatched && !retired[previousAlias] {
			unmatchedRetirements = append(unmatchedRetirements, artifact.AliasBinding{
				Name: previousAlias, Target: document.ID, Previous: artifact.IDPointer(document.ID), Remove: true,
			})
			retired[previousAlias] = true
			aliases.Retired++
			aliases.Preserved--
		}
	}
	if sameStore && !retireUnmatched {
		// Same-store import is recovery, not garbage collection. Retired
		// documents remain discoverable in the current store's document
		// history. Recovery is allowed only when the newest document for its
		// retired alias still matches and every matching historical document
		// agrees on one canonical current document, after alias-event history
		// proves one exact predecessor-to-successor move. Conflicting
		// classifications, explicit retirements, and rollbacks remain retired.
		// Unmatched active decisions stay visible for the stale gate.
		groups := map[string][]closureledger.Document{}
		for _, document := range historicalDocuments {
			if len(document.Bindings) != 1 {
				continue
			}
			previousAlias, err := closureledger.ActiveAlias(document.Bindings[0])
			if err != nil {
				return count, unmatched, first, aliases, err
			}
			if _, active := sourceAliases[previousAlias]; !active {
				groups[previousAlias] = append(groups[previousAlias], document)
			}
		}
		previousAliases := slices.Sorted(maps.Keys(groups))
		type historicalConsensus struct {
			document  closureledger.Document
			ambiguous bool
		}
		consensus := map[string]historicalConsensus{}
		eligible := map[string]bool{}
		blocked := map[string]bool{}
		for _, previousAlias := range previousAliases {
			documents := groups[previousAlias]
			retiredTarget, authorized := recoveryAuthority[previousAlias]
			if !authorized || documents[len(documents)-1].ID != retiredTarget {
				continue
			}
			matches := make([]string, 0, len(documents))
			for _, document := range documents {
				current, matched, _, err := rebindClosure(index, document, reviewCallsites)
				if err != nil {
					return count, unmatched, first, aliases, err
				}
				if !matched {
					continue
				}
				currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
				if err != nil {
					return count, unmatched, first, aliases, err
				}
				matches = append(matches, currentAlias)
				agreement, exists := consensus[currentAlias]
				switch {
				case !exists:
					agreement.document = current
				case agreement.document.ID != current.ID:
					agreement.ambiguous = true
				}
				consensus[currentAlias] = agreement
			}
			latest, latestMatches, _, err := rebindClosure(index, documents[len(documents)-1], reviewCallsites)
			if err != nil {
				return count, unmatched, first, aliases, err
			}
			latestAlias := ""
			if latestMatches {
				latestAlias, err = closureledger.ActiveAlias(latest.Bindings[0])
				if err != nil {
					return count, unmatched, first, aliases, err
				}
				eligible[latestAlias] = true
			}
			for _, matchAlias := range matches {
				if !latestMatches || matchAlias != latestAlias {
					blocked[matchAlias] = true
				}
			}
		}
		aliases := make([]string, 0, len(consensus))
		for alias, agreement := range consensus {
			if eligible[alias] && !blocked[alias] && !agreement.ambiguous && !claimedAliases[alias] {
				aliases = append(aliases, alias)
			}
		}
		slices.Sort(aliases)
		for _, alias := range aliases {
			rebound = append(rebound, consensus[alias].document)
			claimedAliases[alias] = true
		}
	}
	var reboundClaims int
	unmatchedRetirements, reboundClaims, err = excludeReboundClosureRetirements(unmatchedRetirements, rebound)
	if err != nil {
		return count, unmatched, first, aliases, err
	}
	aliases.Retired -= reboundClaims
	aliases.Preserved += reboundClaims
	fixtures, err := closureFixtureImports(context.Background(), source, target, rebound)
	sameView := target == source
	closeErr := source.Close()
	if !sameView {
		closeErr = errors.Join(closeErr, target.Close())
	}
	source, target = nil, nil
	if err != nil || closeErr != nil {
		return count, unmatched, first, aliases, errors.Join(err, closeErr)
	}
	if retireUnmatched && (len(rebound) != 0 || len(retirements) != 0 || len(fixtures) != 0) {
		return count, unmatched, first, aliases, errors.New(
			"closure-scan: retirement requires a settled rebind; run same-store import without -retire-unmatched first",
		)
	}
	if rebound == nil && retirements == nil && unmatchedRetirements == nil {
		return count, unmatched, first, aliases, nil
	}
	operation := closureImportOperation
	if sameStore {
		operation = closureRebindOperation
	}
	commitHead := reviewedTargetHead
	if rebound != nil || retirements != nil || fixtures != nil {
		_, committed, err := commitClosureDocumentsAtHead(
			root, destinationPath, operation, rebound, retirements, fixtures, &commitHead,
		)
		if err != nil {
			return count, unmatched, first, aliases, err
		}
		commitHead = committed
	}
	// Explicit stale retirement is a separate, recognizable authority and
	// deliberately follows recovery. Each removal compares against the exact
	// ID observed at scan time, so concurrent review wins rather than being
	// silently retired.
	if unmatchedRetirements != nil {
		if _, _, err := commitClosureDocumentsAtHead(
			root, destinationPath, closureRetireUnmatchedOperation, nil, unmatchedRetirements, nil, &commitHead,
		); err != nil {
			return count, unmatched, first, aliases, err
		}
	}
	return len(rebound), unmatched, first, aliases, nil
}

// excludeReboundClosureRetirements keeps the recovery and explicit-retirement
// phases disjoint. A live rebound owns its destination alias; retiring the
// pre-rebind target in the following commit would deterministically fail its
// compare-and-set after the rebound has advanced that alias.
func excludeReboundClosureRetirements(
	retirements []artifact.AliasBinding,
	documents []closureledger.Document,
) ([]artifact.AliasBinding, int, error) {
	claimed := make(map[string]struct{}, len(documents))
	for _, document := range documents {
		for _, binding := range document.Bindings {
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return nil, 0, err
			}
			claimed[alias] = struct{}{}
		}
	}
	filtered := make([]artifact.AliasBinding, 0, len(retirements))
	excluded := 0
	for _, retirement := range retirements {
		if _, rebound := claimed[retirement.Name]; rebound {
			excluded++
			continue
		}
		filtered = append(filtered, retirement)
	}
	return filtered, excluded, nil
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

func allClosureDocuments(ctx context.Context, store *overgodb.Store) ([]closureledger.Document, error) {
	documents := make([]closureledger.Document, 0)
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: closureledger.MediaType, Schema: closureledger.Schema,
		}}, Order: overgodb.DocumentOldestFirst,
	}, closureledger.Parse, func(_ overgodb.DocumentView, document closureledger.Document) error {
		documents = append(documents, document)
		return nil
	})
	return documents, err
}

type closureAliasEventHistory struct {
	latest   map[string]overgodb.AliasEvent
	byCommit map[artifact.CommitID][]artifact.AliasBinding
	byAlias  map[string][]overgodb.AliasEvent
	reviewed closureReviewedOperations
}

type closureRecoveryDisposition struct {
	Target            artifact.ID
	Eligible          bool
	Blocker           string
	SuccessorAlias    string
	SuccessorDocument artifact.ID
}

type closureRecoveryAnalysis struct {
	Authorized    map[string]artifact.ID
	Dispositions  map[string]closureRecoveryDisposition
	Reactivations []closureRecoveryReactivation
	History       closureAliasEventHistory
}

type closureRecoveryReactivation struct {
	Alias               string
	RetiredDocument     artifact.ID
	ReactivatedDocument artifact.ID
	CurrentDocument     artifact.ID
	Commit              overgodb.CommitView
	RetirementSequence  uint64
	CurrentSequence     uint64
	Provenance          string
	Eligible            bool
	Blocker             string
}

// closureRecoveryAuthority recognizes only technical alias-move chains that
// end at the exact expected successor still active at the observed store head.
func closureRecoveryAuthority(
	ctx context.Context,
	store *overgodb.Store,
	documents []closureledger.Document,
) (map[string]artifact.ID, error) {
	analysis, err := analyzeClosureRecovery(ctx, store, documents)
	return analysis.Authorized, err
}

func analyzeClosureRecovery(
	ctx context.Context,
	store *overgodb.Store,
	documents []closureledger.Document,
) (closureRecoveryAnalysis, error) {
	analysis := closureRecoveryAnalysis{
		Authorized: map[string]artifact.ID{}, Dispositions: map[string]closureRecoveryDisposition{},
	}
	_, sequence := store.Head()
	if sequence == 0 {
		return analysis, nil
	}
	history, err := readClosureAliasEventHistory(ctx, store, sequence)
	if err != nil {
		return closureRecoveryAnalysis{}, err
	}
	history.reviewed, err = authenticateClosureReviewedOperations(ctx, store, history, documents)
	if err != nil {
		return closureRecoveryAnalysis{}, err
	}
	analysis, err = resolveClosureRecovery(documents, history)
	if err != nil {
		return closureRecoveryAnalysis{}, err
	}
	analysis.Reactivations, err = auditClosureRecoveryReactivations(documents, history)
	return analysis, err
}

func resolveClosureRecovery(
	documents []closureledger.Document,
	history closureAliasEventHistory,
) (closureRecoveryAnalysis, error) {
	analysis := closureRecoveryAnalysis{
		Authorized: map[string]artifact.ID{}, Dispositions: map[string]closureRecoveryDisposition{}, History: history,
	}
	byID := make(map[artifact.ID]closureledger.Document, len(documents))
	for _, document := range documents {
		byID[document.ID] = document
	}
	type node struct {
		alias  string
		target artifact.ID
	}
	memo := map[node]closureRecoveryDisposition{}
	visiting := map[node]bool{}
	var evaluate func(node) (closureRecoveryDisposition, error)
	evaluate = func(current node) (closureRecoveryDisposition, error) {
		if disposition, found := memo[current]; found {
			return disposition, nil
		}
		disposition := closureRecoveryDisposition{Target: current.target}
		if visiting[current] {
			disposition.Blocker = "move-cycle"
			return disposition, nil
		}
		event, found := history.latest[current.alias]
		if !found {
			disposition.Blocker = "no-alias-history"
			memo[current] = disposition
			return disposition, nil
		}
		binding := event.Binding
		if !binding.Remove {
			if binding.Target == current.target {
				disposition.Eligible = true
			} else {
				disposition.Blocker = "conflicting-supersession"
			}
			memo[current] = disposition
			return disposition, nil
		}
		if binding.Previous == nil || binding.Target != *binding.Previous {
			disposition.Blocker = "removal-not-exact-cas"
			memo[current] = disposition
			return disposition, nil
		}
		if binding.Target != current.target {
			disposition.Blocker = "retired-target-mismatch"
			memo[current] = disposition
			return disposition, nil
		}
		if closureOperationKey(event.Commit.Key, string(closureRetireUnmatchedOperation)) {
			disposition.Blocker = "later-explicit-retirement"
			memo[current] = disposition
			return disposition, nil
		}
		if history.reviewed.authenticates(event.Commit, closureRemediateRecoveryOperation) {
			disposition.Blocker = "reviewed-reactivation-remediation"
			memo[current] = disposition
			return disposition, nil
		}
		if history.reviewed.authenticates(event.Commit, closureRestoreReviewedHeadOperation) {
			disposition.Blocker = "reviewed-head-restore"
			memo[current] = disposition
			return disposition, nil
		}
		retired, found := byID[current.target]
		if !found || len(retired.Bindings) != 1 {
			disposition.Blocker = "missing-retired-decision"
			memo[current] = disposition
			return disposition, nil
		}
		retiredAlias, err := closureledger.ActiveAlias(retired.Bindings[0])
		if err != nil {
			return closureRecoveryDisposition{}, err
		}
		if retiredAlias != current.alias {
			disposition.Blocker = "retired-alias-mismatch"
			memo[current] = disposition
			return disposition, nil
		}
		type successor struct {
			alias    string
			document artifact.ID
		}
		var successors []successor
		predecessors := 0
		noncanonical := false
		for _, candidate := range history.byCommit[event.Commit.ID] {
			document, found := byID[candidate.Target]
			if !found || !sameClosureDecision(retired, document) {
				continue
			}
			if len(document.Bindings) != 1 {
				noncanonical = true
				continue
			}
			alias, err := closureledger.ActiveAlias(document.Bindings[0])
			if err != nil {
				return closureRecoveryDisposition{}, err
			}
			if candidate.Name != alias {
				noncanonical = true
				continue
			}
			if candidate.Remove {
				if candidate.Previous == nil || candidate.Target != *candidate.Previous {
					noncanonical = true
					continue
				}
				predecessors++
				continue
			}
			if candidate.Target == retired.ID {
				continue
			}
			successors = append(successors, successor{alias: alias, document: document.ID})
		}
		switch {
		case noncanonical:
			disposition.Blocker = "noncanonical-successor"
		case predecessors != 1:
			disposition.Blocker = "ambiguous-predecessor"
		case len(successors) == 0:
			disposition.Blocker = "missing-successor"
		case len(successors) != 1:
			disposition.Blocker = "ambiguous-successor"
		default:
			next := successors[0]
			disposition.SuccessorAlias = next.alias
			disposition.SuccessorDocument = next.document
			visiting[current] = true
			nextDisposition, err := evaluate(node{alias: next.alias, target: next.document})
			delete(visiting, current)
			if err != nil {
				return closureRecoveryDisposition{}, err
			}
			if nextDisposition.Eligible {
				disposition.Eligible = true
			} else {
				disposition.Blocker = "successor-chain-not-live"
			}
		}
		memo[current] = disposition
		return disposition, nil
	}
	for alias, event := range history.latest {
		if !event.Binding.Remove {
			continue
		}
		disposition, err := evaluate(node{alias: alias, target: event.Binding.Target})
		if err != nil {
			return closureRecoveryAnalysis{}, err
		}
		analysis.Dispositions[alias] = disposition
		if disposition.Eligible {
			analysis.Authorized[alias] = event.Binding.Target
		}
	}
	return analysis, nil
}

func auditClosureRecoveryReactivations(
	documents []closureledger.Document,
	history closureAliasEventHistory,
) ([]closureRecoveryReactivation, error) {
	byID := make(map[artifact.ID]closureledger.Document, len(documents))
	for _, document := range documents {
		byID[document.ID] = document
	}
	aliases := slices.Sorted(maps.Keys(history.byAlias))
	var audits []closureRecoveryReactivation
	for _, alias := range aliases {
		events := history.byAlias[alias]
		if len(events) == 0 || events[len(events)-1].Binding.Remove {
			continue
		}
		var retirement int
		retired := false
		for index := len(events) - 2; index >= 0; index-- {
			if events[index].Binding.Remove {
				retirement = index
				retired = true
				break
			}
		}
		if !retired || retirement+1 >= len(events) {
			continue
		}
		activation := events[retirement+1]
		if history.reviewed.authenticates(activation.Commit, closureRestoreReviewedHeadOperation) {
			continue
		}
		if activation.Binding.Remove || activation.Binding.Previous != nil {
			continue
		}
		expectedAlias, expectedDocument, predecessorFound, err := closureLiveSuccessorBeforeReactivation(
			alias, events[retirement].Binding.Target, activation.Commit.Sequence, history, byID,
		)
		if err != nil {
			return nil, err
		}
		if predecessorFound && closureReactivationHasUniquePredecessor(
			activation, expectedAlias, expectedDocument, history, byID,
		) {
			continue
		}
		retiredDocument, retiredFound := byID[events[retirement].Binding.Target]
		reactivatedDocument, reactivatedFound := byID[activation.Binding.Target]
		audit := closureRecoveryReactivation{
			Alias: alias, RetiredDocument: events[retirement].Binding.Target,
			ReactivatedDocument: activation.Binding.Target,
			CurrentDocument:     events[len(events)-1].Binding.Target, Commit: activation.Commit,
			RetirementSequence: events[retirement].Commit.Sequence,
			CurrentSequence:    events[len(events)-1].Commit.Sequence,
			Provenance:         closureReactivationProvenance(activation.Commit.Key),
		}
		if !retiredFound || !reactivatedFound || !sameClosureDecision(retiredDocument, reactivatedDocument) {
			audit.Blocker = "reactivation-decision-mismatch"
			audits = append(audits, audit)
			continue
		}
		synthetic := history
		synthetic.latest = make(map[string]overgodb.AliasEvent, len(history.latest))
		maps.Copy(synthetic.latest, history.latest)
		synthetic.latest[alias] = events[retirement]
		analysis, err := resolveClosureRecovery(documents, synthetic)
		if err != nil {
			return nil, err
		}
		disposition, found := analysis.Dispositions[alias]
		if !found {
			audit.Blocker = "missing-recovery-disposition"
		} else {
			audit.Eligible = disposition.Eligible
			audit.Blocker = disposition.Blocker
		}
		audits = append(audits, audit)
	}
	return audits, nil
}

func closureReactivationProvenance(key string) string {
	if closureOperationKey(key, string(closureRebindOperation)) {
		return "rebind-operation-unverified"
	}
	return "legacy-operation-unverified"
}

func closureReactivationHasUniquePredecessor(
	activation overgodb.AliasEvent,
	expectedAlias string,
	expectedDocument artifact.ID,
	history closureAliasEventHistory,
	byID map[artifact.ID]closureledger.Document,
) bool {
	activated, found := byID[activation.Binding.Target]
	if !found || len(activated.Bindings) != 1 {
		return false
	}
	alias, err := closureledger.ActiveAlias(activated.Bindings[0])
	if err != nil || alias != activation.Binding.Name {
		return false
	}
	predecessors, successors := 0, 0
	expectedPredecessor := false
	actualSuccessor := false
	for _, binding := range history.byCommit[activation.Commit.ID] {
		document, found := byID[binding.Target]
		if !found || len(document.Bindings) != 1 || !sameClosureDecision(document, activated) {
			continue
		}
		documentAlias, err := closureledger.ActiveAlias(document.Bindings[0])
		if err != nil || documentAlias != binding.Name {
			return false
		}
		if binding.Remove {
			if binding.Previous == nil || binding.Target != *binding.Previous {
				return false
			}
			predecessors++
			expectedPredecessor = expectedPredecessor ||
				binding.Name == expectedAlias && binding.Target == expectedDocument
			continue
		}
		successors++
		actualSuccessor = actualSuccessor ||
			binding.Name == activation.Binding.Name && binding.Target == activation.Binding.Target
	}
	return actualSuccessor && expectedPredecessor && predecessors == 1 && successors == 1
}

func closureLiveSuccessorBeforeReactivation(
	alias string,
	document artifact.ID,
	activationSequence uint64,
	history closureAliasEventHistory,
	byID map[artifact.ID]closureledger.Document,
) (string, artifact.ID, bool, error) {
	before := closureAliasEventHistory{
		latest: map[string]overgodb.AliasEvent{}, byCommit: map[artifact.CommitID][]artifact.AliasBinding{},
		byAlias: map[string][]overgodb.AliasEvent{},
	}
	for name, events := range history.byAlias {
		for _, event := range events {
			if event.Commit.Sequence >= activationSequence {
				break
			}
			before.latest[name] = event
			before.byCommit[event.Commit.ID] = append(before.byCommit[event.Commit.ID], event.Binding)
			before.byAlias[name] = append(before.byAlias[name], event)
		}
	}
	documents := slices.Collect(maps.Values(byID))
	analysis, err := resolveClosureRecovery(documents, before)
	if err != nil {
		return "", artifact.ID{}, false, err
	}
	seen := map[string]bool{}
	for {
		if seen[alias] {
			return "", artifact.ID{}, false, nil
		}
		seen[alias] = true
		event, found := before.latest[alias]
		if !found || event.Binding.Target != document {
			return "", artifact.ID{}, false, nil
		}
		if !event.Binding.Remove {
			return alias, document, true, nil
		}
		disposition, found := analysis.Dispositions[alias]
		if !found || !disposition.Eligible || disposition.SuccessorAlias == "" ||
			!disposition.SuccessorDocument.Valid() {
			return "", artifact.ID{}, false, nil
		}
		alias, document = disposition.SuccessorAlias, disposition.SuccessorDocument
	}
}

func readClosureAliasEventHistory(
	ctx context.Context,
	store *overgodb.Store,
	sequence uint64,
) (closureAliasEventHistory, error) {
	history := closureAliasEventHistory{
		latest: map[string]overgodb.AliasEvent{}, byCommit: map[artifact.CommitID][]artifact.AliasBinding{},
		byAlias: map[string][]overgodb.AliasEvent{},
	}
	if sequence == 0 {
		return history, nil
	}
	err := store.VisitAliasEvents(ctx, overgodb.AliasEventRange{
		Prefix: closureledger.ActiveAliasPrefix, ToSequence: sequence,
	}, func(event overgodb.AliasEvent) error {
		history.latest[event.Binding.Name] = event
		history.byCommit[event.Commit.ID] = append(history.byCommit[event.Commit.ID], event.Binding)
		history.byAlias[event.Binding.Name] = append(history.byAlias[event.Binding.Name], event)
		return nil
	})
	return history, err
}

// closureOperationKey classifies an operation for fail-safe auditing only. A
// matching key is never sufficient authority to recover or reactivate a
// closure; those decisions require the structural history checks above.
func closureOperationKey(key, prefix string) bool {
	suffix := strings.TrimPrefix(key, prefix)
	if suffix == key || len(suffix) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(suffix)
	return err == nil
}

func sameClosureDecision(left, right closureledger.Document) bool {
	return left.Version == right.Version && left.Name == right.Name && bytes.Equal(left.Value, right.Value) &&
		left.Tier == right.Tier && left.Status == right.Status && left.Understanding == right.Understanding &&
		left.ClosurePath == right.ClosurePath && left.RerankTrigger == right.RerankTrigger
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
	owners, commit, err := commitClosureDocuments(
		root, filepath.Join(root, storePath), closurePublishOperation, documents, nil, nil,
	)
	if err != nil {
		return err
	}
	fmt.Printf("emitted %d closure documents (%d owner files) commit %x\n",
		len(documents), owners, commit[:8])
	return nil
}

func commitClosureDocuments(
	root, storePath string,
	operation closureCommitOperation,
	documents []closureledger.Document,
	retirements []artifact.AliasBinding,
	fixtures []artifact.Descriptor,
) (int, artifact.CommitID, error) {
	return commitClosureDocumentsAtHead(root, storePath, operation, documents, retirements, fixtures, nil)
}

func commitClosureDocumentsAtHead(
	root, storePath string,
	operation closureCommitOperation,
	documents []closureledger.Document,
	retirements []artifact.AliasBinding,
	fixtures []artifact.Descriptor,
	expectedHead *artifact.CommitID,
) (int, artifact.CommitID, error) {
	if err := requireUniqueClosureDocumentAliases(documents); err != nil {
		return 0, artifact.CommitID{}, err
	}
	store, err := overgodb.OpenContext(context.Background(), storePath)
	if err != nil {
		return 0, artifact.CommitID{}, err
	}
	defer store.Close()
	batch := artifact.Batch{Aliases: retirements}
	batchedContents := map[artifact.ID]bool{}
	if expectedHead != nil {
		expected := *expectedHead
		batch.ExpectedHead = &expected
	}
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
			// Identical source revisions can repeat one alias in an import. The
			// preflight above refuses distinct decisions for one alias; here an
			// exact live claim replaces a scheduled retirement and duplicates
			// collapse to one compare-and-set operation.
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
	if batch.Empty() {
		head, _ := store.Head()
		if expectedHead != nil && head != *expectedHead {
			return 0, artifact.CommitID{}, fmt.Errorf(
				"closure-scan: reviewed store head moved from %s to %s", *expectedHead, head,
			)
		}
		return len(files), head, nil
	}
	if err := bindClosureOperationKey(operation, &batch); err != nil {
		return 0, artifact.CommitID{}, err
	}
	commit, err := store.Commit(context.Background(), batch)
	if err != nil {
		return 0, artifact.CommitID{}, err
	}
	return len(files), commit, nil
}

func requireUniqueClosureDocumentAliases(documents []closureledger.Document) error {
	claims := map[string]artifact.ID{}
	for _, document := range documents {
		for _, binding := range document.Bindings {
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return err
			}
			if prior, found := claims[alias]; found && prior != document.ID {
				return fmt.Errorf("closure-scan: conflicting reviewed decisions claim alias %s", alias)
			}
			claims[alias] = document.ID
		}
	}
	return nil
}

func bindClosureOperationKey(operation closureCommitOperation, batch *artifact.Batch) error {
	switch operation {
	case closurePublishOperation, closureImportOperation, closureRebindOperation, closureRetireUnmatchedOperation,
		closureRemediateRecoveryOperation, closureRestoreReviewedHeadOperation:
	default:
		return errors.New("closure-scan: invalid commit operation")
	}
	if batch == nil || batch.Key != "" {
		return errors.New("closure-scan: invalid unkeyed operation batch")
	}
	encoded, err := json.Marshal(batch)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(encoded)
	batch.Key = string(operation) + hex.EncodeToString(digest[:])
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
