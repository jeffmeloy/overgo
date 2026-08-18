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
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/discovery"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
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
	refusals := flags.Bool("refusals", false, "list refused decisions with their measured evidence (the refusal ledger)")
	budgets := flags.Bool("budgets", false, "list split partitions and query-budget grants with balances derived from committed charges")
	experiments := flags.Bool("experiments", false, "reconcile experiment lifecycle chains to their current state, flagging expired leases and runners")
	components := flags.Bool("components", false, "list committed component decompositions with per-role counts (classification ledger)")
	proposals := flags.Bool("proposals", false, "list committed bridge proposals with their blocked state, blocker and required verifier")
	admissions := flags.Bool("admissions", false, "list admission bindings: per-generation proposer/evaluator/decider authority domains and prior-generation approvals")
	composed := flags.Bool("composed", false, "list composed model artifacts with their recipes, parents and constituent counts")
	retrieve := flags.String("retrieve", "", "hypervector retrieval: rank the catalog against the named component (lexical organ + distributional signal)")
	rankingID := flags.String("ranking", "", "retrieve: rescore hits with a committed proposal-ranking artifact (the learned prior over the decision ledger)")
	derivations := flags.Bool("derivations", false, "list derivation-profile artifacts and their promotion verdicts (profiles judged by the models they produce)")
	verifications := flags.Bool("verifications", false, "derive the model verification matrix from committed records: strongest evidenced tier per capability")
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
		return writeRetrieve(output, *repository, *retrieve, *rankingID, *limit)
	}
	if *derivations {
		return writeDerivations(output, *repository, *limit)
	}
	if *verifications {
		return writeVerifications(output, *repository, *limit)
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
	budgets := make(map[artifact.ID]runrecord.Budget)
	for _, descriptor := range result.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		if record, err := runrecord.ParseGenerationRecord(content.Data); err == nil {
			records[record.Child] = record
			continue
		}
		if budget, err := runrecord.ParseBudget(content.Data); err == nil {
			budgets[budget.ID] = budget
		}
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
	count := 0
	for _, descriptor := range result.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		decision, err := recipe.ParseDecision(content.Data)
		if err != nil || decision.Outcome != recipe.DecisionRefused {
			continue // other evidence document kinds share the store
		}
		evidence := make([]string, 0, len(decision.Evidence))
		for _, id := range decision.Evidence {
			evidence = append(evidence, id.String())
		}
		fmt.Fprintf(output, "refusal subject=%s tier=%s reason=%q evidence=%s decider=%s\n",
			decision.Subject, decision.Tier, decision.Reason, strings.Join(evidence, ","), decision.Decider.CodeCommit)
		count++
	}
	fmt.Fprintf(output, "%d refusal(s); honesty: rows derive from committed decision documents only; refusals without measurement evidence cannot be committed\n", count)
	return nil
}

// writeExperiments reconciles each experiment's immutable lifecycle chain to
// its current state. Expired leases and runners are reported recoverable with
// their next retry identity -- derived truth, never a mutable status file.
func writeExperiments(output io.Writer, repository string, limit int) error {
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
	chains := map[artifact.ID][]runrecord.ExperimentLifecycle{}
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != runrecord.ExperimentLifecycleMediaType ||
			descriptor.Schema != runrecord.ExperimentLifecycleSchema {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		record, err := runrecord.ParseExperimentLifecycle(content.Data)
		if err != nil {
			return err
		}
		chains[record.Experiment] = append(chains[record.Experiment], record)
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
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindTensorSet, MaxResults: limit})
	if err != nil {
		return err
	}
	count := 0
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != modelartifact.ComponentDecompositionMediaType ||
			descriptor.Schema != modelartifact.ComponentDecompositionSchema {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		decomposition, err := modelartifact.ParseComponentDecomposition(content.Data)
		if err != nil {
			return err
		}
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
	}
	fmt.Fprintf(output, "%d decomposition(s); honesty: rows derive from committed classification documents only; no retrieval index exists yet\n", count)
	return nil
}

// writeProposals lists committed bridge proposals: always promotion-blocked
// with their blocker and required verifier -- the advisory candidate ledger.
func writeProposals(output io.Writer, repository string, limit int) error {
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
	count := 0
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != composition.BridgeProposalMediaType ||
			descriptor.Schema != composition.BridgeProposalSchema {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		proposal, err := composition.ParseBridgeProposal(content.Data)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "proposal %s target=%s candidates=%d state=%s verifier=%q blocker=%q\n",
			proposal.ID, proposal.Target, len(proposal.Candidates), proposal.State,
			proposal.RequiredVerifier, proposal.Blocker)
		count++
	}
	fmt.Fprintf(output, "%d proposal(s); honesty: every row is promotion-blocked by construction; this ledger advises and never authorizes\n", count)
	return nil
}

// writeAdmissions lists admission bindings: which authority domains hold the
// proposer, evaluator and decider roles per generation, and which prior
// approval admitted each succession.
func writeAdmissions(output io.Writer, repository string, limit int) error {
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
	count := 0
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != runrecord.AdmissionBindingMediaType ||
			descriptor.Schema != runrecord.AdmissionBindingSchema {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		binding, err := runrecord.ParseAdmissionBinding(content.Data)
		if err != nil {
			return err
		}
		line := fmt.Sprintf("admission %s generation=%d proposer=%s evaluator=%s decider=%s",
			binding.ID, binding.Generation, binding.Proposer.Name, binding.Evaluator.Name, binding.Decider.Name)
		if binding.PriorApproval.Valid() {
			line += " prior_approval=" + binding.PriorApproval.String()
		}
		fmt.Fprintln(output, line)
		count++
	}
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != runrecord.EvaluatorPromotionMediaType ||
			descriptor.Schema != runrecord.EvaluatorPromotionSchema {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		promotion, err := runrecord.ParseEvaluatorPromotion(content.Data)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "evaluator-promotion %s evaluator=%s oracles=%d promoted=%t reason=%q\n",
			promotion.ID, promotion.Evaluator, len(promotion.Cases), promotion.Promoted, promotion.Reason)
		count++
	}
	fmt.Fprintf(output, "%d admission record(s); honesty: domains must be pairwise distinct by construction; succession and evaluator promotions require the cited prior-authority approvals\n", count)
	return nil
}

// writeComposed lists composed model artifacts: the assembly ledger, each row
// naming the executable recipe, parent models and constituent counts.
func writeComposed(output io.Writer, repository string, limit int) error {
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindModel, MaxResults: limit})
	if err != nil {
		return err
	}
	count := 0
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != composition.ComposedModelMediaType ||
			descriptor.Schema != composition.ComposedModelSchema {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		document, err := composition.ParseComposedModel(content.Data)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "composed %s architecture=%s recipe=%s parents=%d components=%d adapter=%t checkpoint=%t\n",
			document.ID, document.Architecture, document.Recipe, len(document.Parents),
			len(document.Components), document.Adapter.Valid(), document.Checkpoint.Valid())
		count++
	}
	fmt.Fprintf(output, "%d composed artifact(s); honesty: rows derive from committed assembly documents; execution and lineage resolve through the store graph\n", count)
	return nil
}

// writeVerifications derives the model verification matrix from committed
// verification records: one row per model, each capability at the strongest
// evidenced tier. The comparison the compatibility document could never make
// legible, derived from the store, never hand-maintained.
func writeVerifications(output io.Writer, repository string, limit int) error {
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
	records := make([]runrecord.ModelVerification, 0)
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != runrecord.ModelVerificationMediaType {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil || !ok {
			continue
		}
		record, err := runrecord.ParseModelVerification(content.Data)
		if err != nil {
			continue
		}
		records = append(records, record)
	}
	matrix := runrecord.VerificationMatrix(records)
	for _, row := range matrix {
		cells := make([]string, 0, len(row.Capabilities))
		for _, claim := range row.Capabilities {
			cell := fmt.Sprintf("%s=%s(%d evidence,commit=%.12s", claim.Capability, claim.Tier, len(claim.Evidence), claim.Commit)
			if claim.Dataset.Valid() {
				cell += fmt.Sprintf(",dataset=%.20s,steps=%d,tokens=%d", claim.Dataset, claim.SpanSteps, claim.SpanTokens)
			}
			cells = append(cells, cell+")")
		}
		fmt.Fprintf(output, "model %s %s %s\n", row.Name, row.Model, strings.Join(cells, " "))
	}
	fmt.Fprintf(output, "%d model(s) from %d record(s); honesty: rows derive from committed verification records; every tier claim is grounded in named evidence, and unrecorded models simply do not appear\n",
		len(matrix), len(records))
	return nil
}

// writeRetrieve indexes the committed catalog and ranks it against the named
// component. With a ranking artifact, hits are rescored by the learned prior
// trained on the decision ledger; without one, plain similarity stands.
// Advisory output: candidates feed blocked proposals, never promotions.
func writeRetrieve(output io.Writer, repository, name, rankingText string, limit int) error {
	store, err := repodb.OpenReadOnly(repository)
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
	index, err := composition.NewHypervectorIndex(components)
	if err != nil {
		return err
	}
	hits, err := index.Search(*query, limit)
	if err != nil {
		return err
	}
	signalFor := func(hit composition.HypervectorHit) string {
		if hit.Component.Statistics != nil {
			return "lexical+distributional"
		}
		return "lexical"
	}
	if rankingText != "" {
		rankingID, err := artifact.ParseID(rankingText)
		if err != nil {
			return err
		}
		content, ok, err := store.Content(ctx, rankingID)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("ranking %s is not committed", rankingText)
		}
		ranking, err := composition.ParseProposalRanking(content.Data)
		if err != nil {
			return err
		}
		ranked := ranking.Rerank(hits)
		for _, hit := range ranked {
			fmt.Fprintf(output, "hit %.4f relevance=%.4f prior=%+.4f model=%s component=%s role=%s signal=%s\n",
				hit.Score, hit.Relevance, hit.Prior, hit.Component.Model, hit.Component.Name, hit.Component.Contract.Role, signalFor(hit.HypervectorHit))
		}
		fmt.Fprintf(output, "%d hit(s) over %d component(s), signature width %d, prior %s over %d decision(s); honesty: the prior trains only on accepted and refused composition decisions -- advisory retrieval, candidates require blocked proposals and the experiment plane\n",
			len(ranked), index.Len(), composition.HypervectorDimensionsFor(index.Len()), ranking.ID, len(ranking.Sources))
		return nil
	}
	for _, hit := range hits {
		fmt.Fprintf(output, "hit %.4f model=%s component=%s role=%s signal=%s\n",
			hit.Relevance, hit.Component.Model, hit.Component.Name, hit.Component.Contract.Role, signalFor(hit))
	}
	fmt.Fprintf(output, "%d hit(s) over %d component(s), signature width %d; honesty: advisory retrieval, candidates require blocked proposals and the experiment plane\n",
		len(hits), index.Len(), composition.HypervectorDimensionsFor(index.Len()))
	return nil
}

// writeDerivations lists derivation-profile artifacts and their promotion
// verdicts: the policy ledger, where every profile is judged by the models it
// produced.
func writeDerivations(output io.Writer, repository string, limit int) error {
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	count := 0
	profileResult, err := store.Query(ctx, repodb.Query{Kind: artifact.KindProfile, MaxResults: limit})
	if err != nil {
		return err
	}
	for _, descriptor := range profileResult.Artifacts {
		if descriptor.MediaType != scratchmodel.DerivationProfileMediaType {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil || !ok {
			continue
		}
		document, err := scratchmodel.ParseDerivationProfileDocument(content.Data)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "derivation-profile %s version=%s mlp-budget=%d\n",
			document.ID, document.Profile.Version, document.Profile.MLPBudget)
		count++
	}
	promotionResult, err := store.Query(ctx, repodb.Query{Kind: artifact.KindEvidence, MaxResults: limit})
	if err != nil {
		return err
	}
	for _, descriptor := range promotionResult.Artifacts {
		if descriptor.MediaType != scratchmodel.DerivationPromotionMediaType {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil || !ok {
			continue
		}
		promotion, err := scratchmodel.ParseDerivationPromotion(content.Data)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "derivation-promotion %s candidate=%s incumbent=%s promoted=%t reason=%q\n",
			promotion.ID, promotion.Candidate, promotion.Incumbent, promotion.Promoted, promotion.Reason)
		count++
	}
	fmt.Fprintf(output, "%d derivation record(s); honesty: profiles are judged solely by their descendants' measured quality; refusals list beside promotions\n", count)
	return nil
}

// writeBudgets renders split partitions and query-budget grants with balances
// derived from committed charges -- immutable documents only, no counter to
// drift. An exhausted or over-charged grant is reported, never hidden.
func writeBudgets(output io.Writer, repository string, limit int) error {
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
	var grants []runrecord.Budget
	var partitions []runrecord.SplitPartition
	charges := make(map[artifact.ID][]runrecord.BudgetCharge)
	for _, descriptor := range result.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		if partition, err := runrecord.ParseSplitPartition(content.Data); err == nil {
			fmt.Fprintf(output, "partition dataset=%s development=%s selection=%s promotion=%s audit=%s proposer=%s\n",
				partition.Dataset, partition.Development, partition.Selection, partition.Promotion, partition.Audit, partition.Proposer)
			partitions = append(partitions, partition)
			continue
		}
		if budget, err := runrecord.ParseBudget(content.Data); err == nil {
			grants = append(grants, budget)
			continue
		}
		if charge, err := runrecord.ParseBudgetCharge(content.Data); err == nil {
			charges[charge.Budget] = append(charges[charge.Budget], charge)
		}
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
