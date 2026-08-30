// Command overgodb-query reads the catalog: filtered artifact listings,
// lineage traversals, and the derived operational ledgers.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/closurescan"
	"overgo/internal/codemanifest"
	"overgo/internal/composition"
	"overgo/internal/dataset"
	"overgo/internal/discovery"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.Main(func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("overgodb-query", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB root")
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
	refusals := flags.Bool("refusals", false, "list refused decisions with their measured evidence (the refusal ledger)")
	budgets := flags.Bool("budgets", false, "list split partitions and query-budget grants with balances derived from committed charges")
	experiments := flags.Bool("experiments", false, "reconcile experiment lifecycle chains to their current state, flagging expired leases and runners")
	components := flags.Bool("components", false, "list committed component decompositions with per-role counts (classification ledger)")
	proposals := flags.Bool("proposals", false, "list committed bridge proposals with their blocked state, blocker and required verifier")
	admissions := flags.Bool("admissions", false, "list admission bindings: per-generation proposer/evaluator/decider authority domains and prior-generation approvals")
	composed := flags.Bool("composed", false, "list composed model artifacts with their recipes, parents and constituent counts")
	retrieve := flags.String("retrieve", "", "hypervector retrieval: rank the catalog against the named component (lexical organ + distributional signal)")
	verifications := flags.Bool("verifications", false, "derive the model verification matrix from committed records: strongest evidenced tier per capability")
	configs := flags.Bool("configs", false, "list committed model-config declarations: sequence extensions and generation essentials with source digests")
	profiles := flags.Bool("profiles", false, "audit registered architecture-profile publication")
	datasets := flags.Bool("datasets", false, "audit active dataset catalog publication")
	contentDump := flags.Bool("content", false, "print the raw committed content bytes of the artifact named by -id")
	magicClosures := flags.Bool("magic-closures", false, "list magic census history, owner pressure, and unresolved bindings")
	manifestSummary := flags.String("manifest-summary", "", "print a bounded summary for one code-manifest ID")
	if err := flags.Parse(args); err != nil {
		return err
	}
	var emptyResultBound int
	if flags.NArg() != emptyResultBound || strings.TrimSpace(*repository) == "" || *limit <= emptyResultBound {
		return errors.New("usage: overgodb-query -repo <path> [filters]")
	}
	if *servable {
		return writeServable(output, *repository, *limit)
	}
	if *generations {
		return writeGenerations(output, *repository, *limit)
	}
	if *refusals {
		return writeRefusals(output, *repository, *limit)
	}
	if *budgets {
		return writeBudgets(output, *repository, *limit)
	}
	if *experiments {
		return writeExperiments(output, *repository, *limit)
	}
	if *components {
		return writeComponents(output, *repository, *limit)
	}
	if *proposals {
		return writeProposals(output, *repository, *limit)
	}
	if *admissions {
		return writeAdmissions(output, *repository, *limit)
	}
	if *composed {
		return writeComposed(output, *repository, *limit)
	}
	if *retrieve != "" {
		return writeRetrieve(output, *repository, *retrieve, *limit)
	}
	if *verifications {
		return writeVerifications(output, *repository, *limit)
	}
	if *configs {
		return writeConfigs(output, *repository, *limit)
	}
	if *profiles {
		return writeProfiles(output, *repository, *jsonOutput)
	}
	if *datasets {
		return writeDatasets(output, *repository, *jsonOutput)
	}
	if *contentDump {
		return writeContent(output, *repository, *idText)
	}
	if *magicClosures {
		return writeMagicClosures(output, *repository, *limit, *jsonOutput)
	}
	if *manifestSummary != "" {
		return writeManifestSummary(output, *repository, *manifestSummary, *jsonOutput)
	}
	query := overgodb.Query{
		Alias: *alias, MaxDepth: uint32(*maxDepth), MaxResults: *limit,
		FromSequence: *from, ToSequence: *to,
		Projection: overgodb.ProjectCatalog,
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
	store, err := overgodb.OpenReadOnly(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	result, err := store.Query(context.Background(), query)
	if err != nil {
		return err
	}
	if *jsonOutput {
		return clioptions.WritePrettyJSON(output, result)
	}
	return writeText(output, result)
}

func writeManifestSummary(output io.Writer, repository, idText string, jsonOutput bool) error {
	id, err := artifact.ParseID(strings.TrimSpace(idText))
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	var summary codemanifest.Summary
	if manifest, loadErr := codemanifest.Load(context.Background(), store, id); loadErr == nil {
		if summary, err = codemanifest.Summarize(manifest); err != nil {
			return err
		}
	} else if summary, err = codemanifest.LoadDigest(context.Background(), store, id); err != nil {
		// Digest-era manifests store only their footprint; the full
		// content is derivable from git at the recorded source identity.
		return errors.Join(loadErr, err)
	}
	if jsonOutput {
		return clioptions.WritePrettyJSON(output, summary)
	}
	fmt.Fprintf(output, "manifest=%s source=%s analyzer=%s/%s contexts=%d files=%d symbols=%d references=%d external_inputs=%d uncertainty=%d\n",
		summary.ID, summary.SourceIdentity, summary.Analyzer.Name, summary.Analyzer.Version,
		summary.BuildContexts, summary.Files, summary.Symbols, summary.References, summary.ExternalInputs, summary.Uncertainty)
	return nil
}

func writeDatasets(output io.Writer, repository string, jsonOutput bool) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	coverage, err := dataset.InspectCatalog(context.Background(), store)
	if err != nil {
		return err
	}
	if jsonOutput {
		return clioptions.WritePrettyJSON(output, coverage)
	}
	fmt.Fprintf(output, "datasets registered=%d published=%d available=%d complete=%t\n",
		coverage.Registered, coverage.Published, coverage.Available, coverage.Complete)
	for _, entry := range coverage.Entries {
		fmt.Fprintf(output, "dataset=%s status=%s available=%t artifact=%s location=%s\n",
			entry.Entry.Name, entry.Status, entry.Available, entry.Entry.Dataset, entry.Location)
	}
	return nil
}

func writeProfiles(output io.Writer, repository string, jsonOutput bool) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	coverage, err := modelrecipe.InspectArchitectureProfileCatalog(context.Background(), store)
	if err != nil {
		return err
	}
	if jsonOutput {
		return clioptions.WritePrettyJSON(output, coverage)
	}
	fmt.Fprintf(output, "profiles registered=%d published=%d complete=%t\n",
		coverage.Registered, coverage.Published, coverage.Complete)
	for _, entry := range coverage.Entries {
		fmt.Fprintf(output, "architecture=%s status=%s profile=%s\n",
			entry.Architecture, entry.Status, entry.Expected)
	}
	return nil
}

type magicClosureRun struct {
	ID       artifact.ID                `json:"id"`
	Evidence closurescan.CensusEvidence `json:"evidence"`
	Delta    *magicCensusDelta          `json:"delta,omitempty"`
}

type magicCensusDelta struct {
	Counts   closurescan.CensusCounts    `json:"counts"`
	Pressure closurescan.ClosurePressure `json:"closure_pressure"`
}

func visitDocuments[T any](
	ctx context.Context,
	store *overgodb.Store,
	contract artifact.DocumentContract,
	decode func([]byte) (T, error),
	visit func(overgodb.DocumentView, T) error,
) error {
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{contract}, Order: overgodb.DocumentOldestFirst,
	}, decode, visit)
	return err
}

type magicClosureReport struct {
	Runs []magicClosureRun `json:"runs"`
}

func writeMagicClosures(output io.Writer, repository string, limit int, jsonOutput bool) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	var report magicClosureReport
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: closurescan.CensusEvidenceMediaType, Schema: closurescan.CensusEvidenceSchema,
	}, closurescan.ParseCensusEvidence, func(_ overgodb.DocumentView, evidence closurescan.CensusEvidence) error {
		report.Runs = append(report.Runs, magicClosureRun{ID: evidence.ID, Evidence: evidence})
		return nil
	})
	if err != nil {
		return err
	}
	var previous *closurescan.CensusEvidence
	for index := range report.Runs {
		current := &report.Runs[index]
		if previous != nil {
			current.Delta = censusDelta(*previous, current.Evidence)
		}
		previous = &current.Evidence
	}
	if len(report.Runs) > limit {
		report.Runs = report.Runs[len(report.Runs)-limit:]
	}
	if jsonOutput {
		return clioptions.WritePrettyJSON(output, report)
	}
	return writeMagicClosureText(output, report, limit)
}

func censusDelta(previous, current closurescan.CensusEvidence) *magicCensusDelta {
	left, right := previous.Counts, current.Counts
	return &magicCensusDelta{
		Counts: closurescan.CensusCounts{
			ProductionFiles: right.ProductionFiles - left.ProductionFiles, TestFiles: right.TestFiles - left.TestFiles,
			NamedConstants: right.NamedConstants - left.NamedConstants, InlineLiterals: right.InlineLiterals - left.InlineLiterals,
			AssumptionHints: right.AssumptionHints - left.AssumptionHints, TestLiterals: right.TestLiterals - left.TestLiterals,
			TestFixtures: right.TestFixtures - left.TestFixtures, TestAssertions: right.TestAssertions - left.TestAssertions,
			TestPolicyCopies: right.TestPolicyCopies - left.TestPolicyCopies,
			RepeatedGroups:   right.RepeatedGroups - left.RepeatedGroups, RepeatedSites: right.RepeatedSites - left.RepeatedSites,
		},
		Pressure: closurescan.ClosurePressure{
			ActiveDocuments: current.Pressure.ActiveDocuments - previous.Pressure.ActiveDocuments,
			OpenDocuments:   current.Pressure.OpenDocuments - previous.Pressure.OpenDocuments,
			StaleBindings:   current.Pressure.StaleBindings - previous.Pressure.StaleBindings,
		},
	}
}

func writeMagicClosureText(output io.Writer, report magicClosureReport, limit int) error {
	for _, run := range report.Runs {
		counts, pressure := run.Evidence.Counts, run.Evidence.Pressure
		fmt.Fprintf(output, "census=%s source=%s catalog=%s sequence=%d\n", run.ID, run.Evidence.Source, run.Evidence.CatalogHead, run.Evidence.CatalogSequence)
		fmt.Fprintf(output, "counts named=%d inline=%d assumptions=%d test_policy=%d repeated=%d/%d\n",
			counts.NamedConstants, counts.InlineLiterals, counts.AssumptionHints, counts.TestPolicyCopies, counts.RepeatedGroups, counts.RepeatedSites)
		fmt.Fprintf(output, "closures active=%d open=%d stale=%d\n", pressure.ActiveDocuments, pressure.OpenDocuments, pressure.StaleBindings)
		if run.Delta != nil {
			delta := run.Delta
			fmt.Fprintf(output, "delta named=%+d inline=%+d assumptions=%+d test_policy=%+d open=%+d stale=%+d\n",
				delta.Counts.NamedConstants, delta.Counts.InlineLiterals, delta.Counts.AssumptionHints, delta.Counts.TestPolicyCopies,
				delta.Pressure.OpenDocuments, delta.Pressure.StaleBindings)
		}
		for index, owner := range run.Evidence.Owners {
			if index >= limit {
				break
			}
			fmt.Fprintf(output, "owner=%s decision=%d repeated=%d\n", owner.Package, owner.DecisionSurfaces, owner.RepeatedSites)
		}
		for _, row := range run.Evidence.Unresolved {
			fmt.Fprintf(output, "open=%s tier=%s document=%s bindings=%d\n", row.Name, row.Tier, row.Document, len(row.Bindings))
		}
		for _, issue := range run.Evidence.Stale {
			fmt.Fprintf(output, "stale=%s name=%s file=%s\n", issue.Kind, issue.Name, issue.File)
		}
	}
	return nil
}

// writeServable renders the discovery predicate owned by internal/discovery.
func writeServable(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	entries, err := discovery.Servable(context.Background(), store, limit)
	if err != nil {
		return err
	}
	served := 0
	for _, entry := range entries {
		if entry.Stale != "" {
			fmt.Fprintf(output, "stale model=%s location=%s reason=%q\n", entry.Model, entry.Location, entry.Stale)
			continue
		}
		served++
		fmt.Fprintf(output, "servable model=%s tier=%s recipe=%s present=%t location=%s\n",
			entry.Model, entry.Tier, entry.Recipe, entry.Present, entry.Location)
	}
	fmt.Fprintf(output, "%d servable model(s), %d stale activation(s); honesty: every listed file hashes to its recorded component identity; stale activations are reported, never served\n", served, len(entries)-served)
	return nil
}

// writeGenerations lists committed generation records and derives each child's
// descendant-generation depth from the record graph: a model without a record
// is a depth-0 root, and a recorded child is one deeper than its deepest
// parent. Depth is computed here, never stored in a record.
func writeGenerations(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	records := make(map[artifact.ID]runrecord.GenerationRecord)
	budgets := make(map[artifact.ID]runrecord.Budget)
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.GenerationMediaType, Schema: runrecord.GenerationSchema,
	}, runrecord.ParseGenerationRecord, func(_ overgodb.DocumentView, record runrecord.GenerationRecord) error {
		records[record.Child] = record
		return nil
	})
	if err != nil {
		return err
	}
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.BudgetMediaType, Schema: runrecord.BudgetSchema,
	}, runrecord.ParseBudget, func(_ overgodb.DocumentView, budget runrecord.Budget) error {
		budgets[budget.ID] = budget
		return nil
	})
	if err != nil {
		return err
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
		if budget, ok := budgets[record.Budget]; ok {
			if err := runrecord.ValidateGenerationBudget(record, budget); err != nil {
				fmt.Fprintf(output, "generation child=%s BUDGET VIOLATION: %v\n", child, err)
			}
		}
	}
	fmt.Fprintf(output, "%d generation record(s); honesty: depth derives from committed generation records only; models without records are depth-0 roots; seed consumption validates against committed budget grants when present\n", len(records))
	return nil
}

// writeRefusals is the refusal ledger: refused decisions read straight from
// committed decision documents. There is no separate ledger document to drift
// -- the store's decision lifecycle is the single source, and every row names
// the measurement evidence that grounded the refusal.
func writeRefusals(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	count := 0
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: recipe.DecisionMediaType, Schema: recipe.DecisionSchema,
	}, recipe.ParseDecision, func(_ overgodb.DocumentView, decision recipe.Decision) error {
		if decision.Outcome != recipe.DecisionRefused {
			return nil
		}
		evidence := make([]string, 0, len(decision.Evidence))
		for _, id := range decision.Evidence {
			evidence = append(evidence, id.String())
		}
		fmt.Fprintf(output, "refusal subject=%s tier=%s reason=%q evidence=%s decider=%s\n",
			decision.Subject, decision.Tier, decision.Reason, strings.Join(evidence, ","), decision.Decider.CodeCommit)
		count++
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "%d refusal(s); honesty: rows derive from committed decision documents only; refusals without measurement evidence cannot be committed\n", count)
	return nil
}

// writeExperiments reconciles each experiment's immutable lifecycle chain to
// its current state. Expired leases and runners are reported recoverable with
// their next retry identity -- derived truth, never a mutable status file.
func writeExperiments(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	chains := map[artifact.ID][]runrecord.ExperimentLifecycle{}
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.ExperimentLifecycleMediaType, Schema: runrecord.ExperimentLifecycleSchema,
	}, runrecord.ParseExperimentLifecycle, func(_ overgodb.DocumentView, record runrecord.ExperimentLifecycle) error {
		chains[record.Experiment] = append(chains[record.Experiment], record)
		return nil
	})
	if err != nil {
		return err
	}
	ids := make([]string, 0, len(chains))
	byText := map[string]artifact.ID{}
	for id := range chains {
		text := id.String()
		ids = append(ids, text)
		byText[text] = id
	}
	sort.Strings(ids)
	now := time.Now()
	for _, text := range ids {
		status, err := runrecord.ReconcileExperiment(chains[byText[text]], now)
		if err != nil {
			return err
		}
		line := fmt.Sprintf("experiment %s state=%s retry=%d records=%d",
			text, status.Tip.State, status.Tip.Retry, len(chains[byText[text]]))
		if status.Expired {
			line += fmt.Sprintf(" EXPIRED heartbeat=%s next_retry=%d", status.Tip.HeartbeatExpiry, status.NextRetry)
		}
		fmt.Fprintln(output, line)
	}
	fmt.Fprintf(output, "%d experiment(s); honesty: state derives from committed lifecycle chains only; a diverged or illegal chain errors rather than guesses\n", len(ids))
	return nil
}

// writeComponents lists committed component decompositions with per-role
// counts: the classification ledger, read from immutable documents only.
func writeComponents(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	count := 0
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindTensorSet, MediaType: modelartifact.ComponentDecompositionMediaType, Schema: modelartifact.ComponentDecompositionSchema,
	}, modelartifact.ParseComponentDecomposition, func(_ overgodb.DocumentView, decomposition modelartifact.ComponentDecompositionDocument) error {
		roles := map[string]int{}
		for _, component := range decomposition.Components {
			roles[string(component.Contract.Role)]++
		}
		names := make([]string, 0, len(roles))
		for role := range roles {
			names = append(names, role)
		}
		sort.Strings(names)
		summary := make([]string, 0, len(names))
		for _, role := range names {
			summary = append(summary, fmt.Sprintf("%s=%d", role, roles[role]))
		}
		fmt.Fprintf(output, "decomposition %s model=%s components=%d %s\n",
			decomposition.ID, decomposition.Model, len(decomposition.Components), strings.Join(summary, " "))
		count++
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "%d decomposition(s); honesty: rows derive from committed classification documents only; no retrieval index exists yet\n", count)
	return nil
}

// writeProposals lists committed bridge proposals: always promotion-blocked
// with their blocker and required verifier -- the advisory candidate ledger.
func writeProposals(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	count := 0
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: composition.BridgeProposalMediaType, Schema: composition.BridgeProposalSchema,
	}, composition.ParseBridgeProposal, func(_ overgodb.DocumentView, proposal composition.BridgeProposal) error {
		fmt.Fprintf(output, "proposal %s target=%s candidates=%d state=%s verifier=%q blocker=%q\n",
			proposal.ID, proposal.Target, len(proposal.Candidates), proposal.State,
			proposal.RequiredVerifier, proposal.Blocker)
		count++
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "%d proposal(s); honesty: every row is promotion-blocked by construction; this ledger advises and never authorizes\n", count)
	return nil
}

// writeAdmissions lists admission bindings: which authority domains hold the
// proposer, evaluator and decider roles per generation, and which prior
// approval admitted each succession.
func writeAdmissions(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	count := 0
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.AdmissionBindingMediaType, Schema: runrecord.AdmissionBindingSchema,
	}, runrecord.ParseAdmissionBinding, func(_ overgodb.DocumentView, binding runrecord.AdmissionBinding) error {
		line := fmt.Sprintf("admission %s generation=%d proposer=%s evaluator=%s decider=%s",
			binding.ID, binding.Generation, binding.Proposer.Name, binding.Evaluator.Name, binding.Decider.Name)
		if binding.PriorApproval.Valid() {
			line += " prior_approval=" + binding.PriorApproval.String()
		}
		fmt.Fprintln(output, line)
		count++
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "%d admission binding(s); honesty: domains must be pairwise distinct by construction; succession requires the cited prior-authority approval\n", count)
	return nil
}

// writeComposed lists composed model artifacts: the assembly ledger, each row
// naming the executable recipe, parent models and constituent counts.
func writeComposed(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	count := 0
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindModel, MediaType: composition.ComposedModelMediaType, Schema: composition.ComposedModelSchema,
	}, composition.ParseComposedModel, func(_ overgodb.DocumentView, document composition.ComposedModelDocument) error {
		fmt.Fprintf(output, "composed %s architecture=%s recipe=%s parents=%d components=%d adapter=%t checkpoint=%t\n",
			document.ID, document.Architecture, document.Recipe, len(document.Parents),
			len(document.Components), document.Adapter.Valid(), document.Checkpoint.Valid())
		count++
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "%d composed artifact(s); honesty: rows derive from committed assembly documents; execution and lineage resolve through the store graph\n", count)
	return nil
}

// writeConfigs lists committed model-config declarations: the typed
// inference- and training-relevant components generic code reads instead of
// carrying model-specific literals.
func writeContent(output io.Writer, repository, idText string) error {
	id, err := artifact.ParseID(idText)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	content, ok, err := artifact.ReadContent(context.Background(), store, id)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("overgodb-query: no committed content for %s", id)
	}
	_, err = output.Write(content.Data)
	return err
}

func writeConfigs(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	count := 0
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: modelartifact.ModelConfigMediaType, Schema: modelartifact.ModelConfigSchema,
	}, modelartifact.ParseModelConfigDocument, func(_ overgodb.DocumentView, document modelartifact.ModelConfigDocument) error {
		var cells []string
		if document.Sequence != nil {
			cells = append(cells, fmt.Sprintf("sequence[k=%d range=[%d,%d) specials=%d auto=%t]",
				document.Sequence.K, document.Sequence.StartID,
				document.Sequence.StartID+document.Sequence.Vocabulary,
				len(document.Sequence.SpecialTokens), document.Sequence.AutoTags))
		}
		if document.Generation != nil {
			cells = append(cells, fmt.Sprintf("generation[bos=%v eos=%v context=%d]",
				document.Generation.BOSTokens, document.Generation.EOSTokens, document.Generation.ContextLength))
		}
		names := make([]string, 0, len(document.Sources))
		for _, source := range document.Sources {
			names = append(names, source.Name)
		}
		fmt.Fprintf(output, "config model=%s %s sources=%s\n",
			document.Model, strings.Join(cells, " "), strings.Join(names, ","))
		count++
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "%d model-config declaration(s); honesty: components derive from digested source files committed with model lineage; generic code reads these declarations, never literals\n", count)
	return nil
}

// writeVerifications derives the model verification matrix from committed
// verification records: one row per model, each capability at the strongest
// evidenced tier. The comparison the compatibility document could never make
// legible, derived from the store, never hand-maintained.
func writeVerifications(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	records := make([]runrecord.ModelVerification, 0)
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.ModelVerificationMediaType, Schema: runrecord.ModelVerificationSchema,
	}, runrecord.ParseModelVerification, func(_ overgodb.DocumentView, record runrecord.ModelVerification) error {
		records = append(records, record)
		return nil
	})
	if err != nil {
		return err
	}
	matrix := runrecord.VerificationMatrix(records)
	if limit > 0 && len(matrix) > limit {
		matrix = matrix[:limit]
	}
	for _, row := range matrix {
		cells := make([]string, 0, len(row.Capabilities))
		for _, claim := range row.Capabilities {
			cell := fmt.Sprintf("%s=%s(%d evidence,commit=%.12s", claim.Capability, claim.Tier, len(claim.Evidence), claim.Commit)
			if claim.Dataset.Valid() {
				cell += fmt.Sprintf(",dataset=%.20s,steps=%d,tokens=%d", claim.Dataset, claim.SpanSteps, claim.SpanTokens)
			}
			if claim.WallNS > 0 {
				cell += fmt.Sprintf(",wall=%s", time.Duration(claim.WallNS).Round(time.Millisecond))
				if claim.ContextTokens > 0 {
					cell += fmt.Sprintf(",ctx=%d", claim.ContextTokens)
				}
				if claim.PeakDeviceBytes > 0 {
					cell += fmt.Sprintf(",peak=%.1fGiB", float64(claim.PeakDeviceBytes)/(1<<30))
				}
			}
			cells = append(cells, cell+")")
		}
		fmt.Fprintf(output, "model %s %s %s\n", row.Name, row.Model, strings.Join(cells, " "))
	}
	fmt.Fprintf(output, "%d model(s) from %d record(s); honesty: rows derive from committed verification records; every tier claim is grounded in named evidence, and unrecorded models simply do not appear\n",
		len(matrix), len(records))
	return nil
}

// writeRetrieve ranks the committed component catalog.
func writeRetrieve(output io.Writer, repository, name string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	components, err := composition.LoadCatalog(ctx, store)
	if err != nil {
		return err
	}
	var query *composition.CatalogComponent
	for i := range components {
		if components[i].Name == name {
			query = &components[i]
			break
		}
	}
	if query == nil {
		return fmt.Errorf("component %q is not in the committed catalog", name)
	}
	index, err := composition.NewExactComponentIndex(components)
	if err != nil {
		return err
	}
	hits, err := index.Search(*query, limit)
	if err != nil {
		return err
	}
	signalFor := func(hit composition.ExactHit) string {
		if hit.Descriptor.Measured {
			return "lexical+distributional"
		}
		return "lexical"
	}
	for _, hit := range hits {
		fmt.Fprintf(output, "hit %.4f model=%s component=%s role=%s signal=%s\n",
			hit.Relevance, hit.Descriptor.Model, hit.Descriptor.Name, hit.Descriptor.Role, signalFor(hit))
	}
	fmt.Fprintf(output, "%d hit(s) over %d component(s), exact inverted index; honesty: advisory retrieval, candidates require blocked proposals and the experiment plane\n",
		len(hits), index.Len())
	return nil
}

// writeBudgets renders split partitions and query-budget grants with balances
// derived from committed charges -- immutable documents only, no counter to
// drift. An exhausted or over-charged grant is reported, never hidden.
func writeBudgets(output io.Writer, repository string, limit int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	var grants []runrecord.Budget
	var partitions []runrecord.SplitPartition
	charges := make(map[artifact.ID][]runrecord.BudgetCharge)
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.SplitPartitionMediaType, Schema: runrecord.SplitPartitionSchema,
	}, runrecord.ParseSplitPartition, func(_ overgodb.DocumentView, partition runrecord.SplitPartition) error {
		fmt.Fprintf(output, "partition dataset=%s development=%s selection=%s promotion=%s audit=%s proposer=%s\n",
			partition.Dataset, partition.Development, partition.Selection, partition.Promotion, partition.Audit, partition.Proposer)
		partitions = append(partitions, partition)
		return nil
	})
	if err != nil {
		return err
	}
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.BudgetMediaType, Schema: runrecord.BudgetSchema,
	}, runrecord.ParseBudget, func(_ overgodb.DocumentView, budget runrecord.Budget) error {
		grants = append(grants, budget)
		return nil
	})
	if err != nil {
		return err
	}
	err = visitDocuments(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.BudgetChargeMediaType, Schema: runrecord.BudgetChargeSchema,
	}, runrecord.ParseBudgetCharge, func(_ overgodb.DocumentView, charge runrecord.BudgetCharge) error {
		charges[charge.Budget] = append(charges[charge.Budget], charge)
		return nil
	})
	if err != nil {
		return err
	}
	for _, budget := range grants {
		remaining, err := runrecord.BudgetBalance(budget, charges[budget.ID])
		if err != nil {
			fmt.Fprintf(output, "budget split=%s unit=%s issued=%d INVALID: %v\n", budget.Split, budget.Unit, budget.Issued, err)
			continue
		}
		fmt.Fprintf(output, "budget split=%s unit=%s issued=%d charged=%d remaining=%d\n",
			budget.Split, budget.Unit, budget.Issued, budget.Issued-remaining, remaining)
		for _, partition := range partitions {
			if err := runrecord.ValidateBlinding(partition, budget, charges[budget.ID]); err != nil {
				fmt.Fprintf(output, "budget split=%s BLINDING VIOLATION: %v\n", budget.Split, err)
			}
		}
	}
	fmt.Fprintf(output, "%d partition(s), %d budget(s); honesty: balances and blinding derive from committed documents only; violations and over-consumption report loudly rather than clamping\n",
		len(partitions), len(grants))
	return nil
}

func parseFollow(value string) (overgodb.FollowDirection, error) {
	switch value {
	case "none":
		return overgodb.FollowNone, nil
	case "parents":
		return overgodb.FollowParents, nil
	case "children":
		return overgodb.FollowChildren, nil
	case "both":
		return overgodb.FollowBoth, nil
	default:
		return overgodb.FollowNone, fmt.Errorf("overgodb-query: invalid follow direction %q", value)
	}
}

func writeText(output io.Writer, result overgodb.QueryResult) error {
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
