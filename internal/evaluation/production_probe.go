package evaluation

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/invocation"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const (
	// CapabilityProbeResultMediaType identifies exact production-path probe evidence.
	CapabilityProbeResultMediaType = "application/vnd.overgo.capability-probe-result+json"
	// CapabilityProbeResultSchema identifies the production-path probe contract.
	CapabilityProbeResultSchema = "overgo/capability-probe-result/v1"
)

// ProductionCapabilityEntry is the closed set of entry points that can
// produce activation evidence. Direct executor and adapter calls are not
// production entries and cannot be represented here.
type ProductionCapabilityEntry string

const (
	// ProductionEntryLocalPlacement is execution through an exact local placement.
	ProductionEntryLocalPlacement ProductionCapabilityEntry = "local-placement"
	// ProductionEntryPeerPlacement is execution through an exact peer placement.
	ProductionEntryPeerPlacement ProductionCapabilityEntry = "peer-placement"
	// ProductionEntryAgentCoordinator is an agent invocation admitted by the coordinator.
	ProductionEntryAgentCoordinator ProductionCapabilityEntry = "agent-coordinator"
)

// CapabilityProbeEvidence supplies existing owner values to one probe
// publication. Selection and placement remain derived identities rather than
// becoming a second lifecycle owned by evaluation.
type CapabilityProbeEvidence struct {
	Case          artifact.ID                                `json:"case"`
	Profile       artifact.ID                                `json:"profile"`
	Placement     capabilityruntime.ExactCapabilityPlacement `json:"placement"`
	Environment   artifact.ID                                `json:"environment"`
	Trace         artifact.ID                                `json:"trace"`
	StageReceipt  artifact.ID                                `json:"stage_receipt"`
	CheckDecision artifact.ID                                `json:"check_decision"`
	Observation   artifact.ID                                `json:"observation"`
}

// CapabilityProbeResult binds one activation case to the exact current model
// execution and the typed records emitted by a production entry point.
type CapabilityProbeResult struct {
	Version         uint16                       `json:"version"`
	Case            artifact.ID                  `json:"case"`
	Profile         artifact.ID                  `json:"profile"`
	Entry           ProductionCapabilityEntry    `json:"entry"`
	Model           artifact.ID                  `json:"model"`
	Recipe          artifact.ID                  `json:"recipe"`
	Capability      artifact.ID                  `json:"capability"`
	ModelCapability artifact.ID                  `json:"model_capability"`
	Selection       artifact.ID                  `json:"selection"`
	Placement       artifact.ID                  `json:"placement"`
	Alias           string                       `json:"alias"`
	Session         modelrecipe.SessionSelection `json:"session"`
	Compatibility   artifact.ID                  `json:"compatibility,omitzero"`
	Environment     artifact.ID                  `json:"environment"`
	Trace           artifact.ID                  `json:"trace"`
	StageReceipt    artifact.ID                  `json:"stage_receipt"`
	CheckDecision   artifact.ID                  `json:"check_decision"`
	Observation     artifact.ID                  `json:"observation"`
	Outcome         runrecord.Outcome            `json:"outcome"`
	Resources       runrecord.ResourceFitness    `json:"resources"`
	ID              artifact.ID                  `json:"-"`
}

var capabilityProbeResultCodec = artifact.JSONDocumentCodec(
	"capability probe result", artifact.KindEvidence,
	CapabilityProbeResultMediaType, CapabilityProbeResultSchema,
	canonicalizeCapabilityProbeResult,
	func(value CapabilityProbeResult) artifact.ID { return value.ID },
	func(value *CapabilityProbeResult, id artifact.ID) { value.ID = id },
	cloneCapabilityProbeResult,
)

// PublishProductionCapabilityProbe validates every referenced owner at the
// current repository head and commits one content-addressed evidence record.
// It advances no alias and owns no activation lifecycle.
func PublishProductionCapabilityProbe(
	ctx context.Context,
	repository artifact.Repository,
	evidence CapabilityProbeEvidence,
) (CapabilityProbeResult, artifact.CommitID, error) {
	if ctx == nil || repository == nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, errors.New("evaluation: capability probe authority is absent")
	}
	head, _ := repository.Head()
	profile, err := activationProfileCodec.Require(ctx, repository, evidence.Profile)
	if err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	selection := evidence.Placement.Selection
	definition := selection.Program.Definition()
	receipt, err := requireCurrentStageReceipt(ctx, repository, evidence.StageReceipt)
	if err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	outcome, err := stageReceiptOutcome(receipt)
	if err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	entry, err := productionEntry(profile, selection, receipt)
	if err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	observation, resources, err := requireProbeResourceObservation(
		ctx, repository, evidence.Observation, evidence.Placement.Capability.ID,
	)
	if err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	result, err := capabilityProbeResultCodec.New(CapabilityProbeResult{
		Version: artifact.InitialDocumentVersion,
		Case:    evidence.Case, Profile: profile.ID, Entry: entry,
		Model: definition.Model, Recipe: definition.ID, Capability: profile.Capability,
		ModelCapability: evidence.Placement.Capability.ID,
		Selection:       selection.Identity, Placement: evidence.Placement.ID,
		Alias: selection.Alias, Session: selection.Session, Compatibility: selection.Peer.ID,
		Environment: evidence.Environment, Trace: evidence.Trace, StageReceipt: receipt.ID,
		CheckDecision: evidence.CheckDecision, Observation: observation.ID,
		Outcome: outcome, Resources: resources,
	})
	if err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	if err := validateCapabilityProbeReferences(ctx, repository, result); err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	if existing, found, readErr := capabilityProbeResultCodec.Read(ctx, repository, result.ID); readErr != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, readErr
	} else if found {
		if err := validateCapabilityProbeReferences(ctx, repository, existing); err != nil {
			return CapabilityProbeResult{}, artifact.CommitID{}, err
		}
		commit, _ := repository.Head()
		return existing, commit, nil
	}
	content, err := capabilityProbeResultCodec.Content(result)
	if err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/capability-probe/"+result.ID.String(),
		[]artifact.Content{content}, result.Lineage(), nil,
	)
	if err != nil {
		return CapabilityProbeResult{}, artifact.CommitID{}, err
	}
	batch.ExpectedHead = &head
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	return result, commit, err
}

// RequireCapabilityProbeResult loads one probe and revalidates its active
// selection, exact placement, and current terminal records. Stale evidence is
// not usable as current activation evidence.
func RequireCapabilityProbeResult(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (CapabilityProbeResult, error) {
	result, err := capabilityProbeResultCodec.RequireExactLineage(ctx, reader, id, CapabilityProbeResult.Lineage)
	if err != nil {
		return CapabilityProbeResult{}, err
	}
	if err := validateCapabilityProbeReferences(ctx, reader, result); err != nil {
		return CapabilityProbeResult{}, err
	}
	return result, nil
}

// Lineage returns only durable owner documents. Selection and placement are
// derived identities recomputed during validation, not synthetic documents.
func (value CapabilityProbeResult) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Case, value.Profile, value.Model, value.Recipe, value.Capability,
		value.ModelCapability, value.Environment, value.Trace, value.StageReceipt,
		value.CheckDecision, value.Observation,
	}
	parents = append(parents, value.Resources.Authorities()...)
	seen := make(map[artifact.ID]struct{}, len(parents))
	unique := make([]artifact.ID, 0, len(parents))
	for _, parent := range parents {
		if !parent.Valid() || parent == value.ID {
			continue
		}
		if _, duplicate := seen[parent]; duplicate {
			continue
		}
		seen[parent] = struct{}{}
		unique = append(unique, parent)
	}
	return artifact.DependencyLineage(value.ID, unique...)
}

func canonicalizeCapabilityProbeResult(value *CapabilityProbeResult) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !value.Entry.valid() ||
		value.Case.Kind() != artifact.KindRecipe || value.Profile.Kind() != artifact.KindProfile ||
		value.Model.Kind() != artifact.KindModel || value.Recipe.Kind() != artifact.KindRecipe ||
		value.Capability.Kind() != artifact.KindProfile || value.ModelCapability.Kind() != artifact.KindProfile ||
		value.Selection.Kind() != artifact.KindProfile || value.Placement.Kind() != artifact.KindProfile ||
		value.Environment.Kind() != artifact.KindEvidence || value.Trace.Kind() != artifact.KindEvidence ||
		value.StageReceipt.Kind() != artifact.KindEvidence || value.CheckDecision.Kind() != artifact.KindEvidence ||
		value.Observation.Kind() != artifact.KindEvidence || !value.Session.Valid() ||
		value.Alias == "" || strings.TrimSpace(value.Alias) != value.Alias || len(value.Alias) > artifact.MaxContentBytes ||
		strings.ContainsAny(value.Alias, "\x00\r\n\t") ||
		value.Compatibility.Valid() && value.Compatibility.Kind() != artifact.KindEvidence ||
		value.Session == modelrecipe.SessionSpillover != value.Compatibility.Valid() ||
		value.Outcome != runrecord.OutcomeSucceeded && value.Outcome != runrecord.OutcomeFailed {
		return errors.New("evaluation: invalid capability probe result")
	}
	if err := value.Resources.Validate(); err != nil {
		return errors.Join(errors.New("evaluation: invalid capability probe resources"), err)
	}
	scope := value.Resources.Scope
	if scope.Model != value.Model || scope.Hardware != value.Environment ||
		scope.Provider != value.ModelCapability || scope.Workload != value.Recipe ||
		scope.Attempt != value.Observation || scope.Surface != runrecord.SurfaceServing {
		return errors.New("evaluation: capability probe resource authorities differ")
	}
	if _, measured := value.Resources.Measure(runrecord.ResourceWallNS); !measured {
		return errors.New("evaluation: capability probe lacks measured wall time")
	}
	return nil
}

func validateCapabilityProbeReferences(ctx context.Context, reader artifact.Reader, result CapabilityProbeResult) error {
	if ctx == nil || reader == nil {
		return errors.New("evaluation: capability probe reader is absent")
	}
	if err := capabilityProbeResultCodec.ValidateIdentity(result); err != nil {
		return err
	}
	testCase, err := activationCaseCodec.Require(ctx, reader, result.Case)
	if err != nil {
		return err
	}
	profile, err := activationProfileCodec.Require(ctx, reader, result.Profile)
	if err != nil || profile.Capability != result.Capability || !slices.Contains(profile.Tasks, testCase.Task) {
		return errors.Join(errors.New("evaluation: capability probe case and profile differ"), err)
	}
	profileCapability, err := runrecord.RequireCapabilityIdentity(ctx, reader, profile.Capability)
	if err != nil {
		return err
	}
	selection, err := modelrecipe.ResolveCapabilityEvidenceSelector(ctx, reader, modelrecipe.CapabilityEvidenceSelector{
		Alias: result.Alias, Task: testCase.Task, Session: result.Session,
		Compatibility: result.Compatibility,
	})
	if err != nil || selection.Identity != result.Selection {
		return errors.Join(errors.New("evaluation: capability probe selection is stale or differs"), err)
	}
	placement, err := capabilityruntime.ResolveExactCapabilityPlacement(ctx, reader, result.ModelCapability, selection)
	if err != nil || placement.ID != result.Placement {
		return errors.Join(errors.New("evaluation: capability probe placement is stale or differs"), err)
	}
	definition := selection.Program.Definition()
	if definition.Model != result.Model || definition.ID != result.Recipe || definition.Task != testCase.Task ||
		placement.Capability.ID != result.ModelCapability {
		return errors.New("evaluation: capability probe model execution differs")
	}
	environment, err := runrecord.RequireEnvironment(ctx, reader, result.Environment)
	if err != nil {
		return err
	}
	if err := validateProbeEnvironment(environment, placement); err != nil {
		return err
	}
	trace, err := runrecord.RequireInteractionTrace(ctx, reader, result.Trace)
	if err != nil || trace.Model != result.Model || trace.Recipe != result.Recipe {
		return errors.Join(errors.New("evaluation: capability probe trace differs"), err)
	}
	if _, err := runrecord.RequireInteractionTranscript(ctx, reader, trace.Request); err != nil {
		return errors.Join(errors.New("evaluation: capability probe request transcript differs"), err)
	}
	receipt, err := requireCurrentStageReceipt(ctx, reader, result.StageReceipt)
	if err != nil || receipt.Recipe != result.Recipe || !programContainsNode(selection, receipt.Node) ||
		!slices.Contains(trace.FinalArtifacts, receipt.ID) || !stageReceiptMatchesOutcome(receipt, result.Outcome) {
		return errors.Join(errors.New("evaluation: capability probe terminal execution differs"), err)
	}
	decision, err := recipe.RequireDecision(ctx, reader, result.CheckDecision)
	if err != nil || decision.Subject != testCase.ID || decision.Decider.Derivation != testCase.EvidenceCheck ||
		decision.Tier != recipe.EvidenceProduction || !probeDecisionMatchesOutcome(decision.Outcome, result.Outcome) ||
		!slices.Contains(decision.Evidence, result.Trace) || !slices.Contains(decision.Evidence, result.StageReceipt) ||
		!slices.Contains(decision.Evidence, result.Observation) || !slices.Contains(decision.Evidence, testCase.Input) {
		return errors.Join(errors.New("evaluation: activation evidence check decision differs"), err)
	}
	observation, resources, err := requireProbeResourceObservation(ctx, reader, result.Observation, result.ModelCapability)
	if err != nil || observation.Model != result.Model || observation.Recipe != result.Recipe ||
		observation.Environment != result.Environment || observation.Operation != receipt.Operation ||
		observation.Task != testCase.Task || observation.Outcome != result.Outcome ||
		!sameResourceFitness(resources, result.Resources) {
		return errors.Join(errors.New("evaluation: capability probe serving observation differs"), err)
	}
	if err := validateProbeResourceEnvelope(result.Resources, placement.Capability.Resources); err != nil {
		return err
	}
	switch result.Entry {
	case ProductionEntryLocalPlacement:
		if profile.Kind != ActivationProfileLocalModel || profileCapability.ID != placement.Capability.ID ||
			selection.Session == modelrecipe.SessionSpillover || receipt.Invocation != nil ||
			trace.Operation != receipt.Operation ||
			trace.TaskContract.Valid() && trace.TaskContract != testCase.Contract ||
			trace.Terminal != "" && trace.Terminal != result.Outcome {
			return errors.New("evaluation: local capability probe bypasses its production placement")
		}
		if calls, _, exchangeErr := trace.ToolExchanges(); exchangeErr != nil || len(calls) != 0 {
			return errors.Join(errors.New("evaluation: local capability probe contains an agent adapter path"), exchangeErr)
		}
	case ProductionEntryPeerPlacement:
		if profile.Kind != ActivationProfilePeerModel || profileCapability.ID != placement.Capability.ID ||
			selection.Session != modelrecipe.SessionSpillover || receipt.Invocation != nil ||
			trace.Operation != receipt.Operation ||
			trace.TaskContract.Valid() && trace.TaskContract != testCase.Contract ||
			trace.Terminal != "" && trace.Terminal != result.Outcome {
			return errors.New("evaluation: peer capability probe bypasses its production placement")
		}
		if calls, _, exchangeErr := trace.ToolExchanges(); exchangeErr != nil || len(calls) != 0 {
			return errors.Join(errors.New("evaluation: peer capability probe contains an agent adapter path"), exchangeErr)
		}
	case ProductionEntryAgentCoordinator:
		if profile.Kind != ActivationProfileAgentTool || result.Outcome != runrecord.OutcomeSucceeded ||
			trace.Operation.Valid() || trace.Terminal != "" && trace.Terminal != result.Outcome ||
			trace.TaskContract.Valid() && trace.TaskContract != testCase.Contract {
			return errors.New("evaluation: agent capability probe differs from its coordinator path")
		}
		if err := validateAgentCoordinatorProbe(ctx, reader, profileCapability, trace, receipt); err != nil {
			return err
		}
	default:
		return errors.New("evaluation: foreign capability probe production entry")
	}
	return nil
}

func validateProbeEnvironment(
	environment runrecord.Environment,
	placement capabilityruntime.ExactCapabilityPlacement,
) error {
	if placement.Selection.Session == modelrecipe.SessionSpillover {
		if environment.ID != placement.Selection.Peer.PeerEnvironment {
			return errors.New("evaluation: peer capability probe environment differs")
		}
		return nil
	}
	if environment.OS != placement.Capability.Platform.OS || environment.Arch != placement.Capability.Platform.Arch {
		return errors.New("evaluation: local capability probe environment differs")
	}
	return nil
}

func validateProbeResourceEnvelope(resources runrecord.ResourceFitness, envelope runrecord.CapabilityResourceEnvelope) error {
	for _, bound := range []struct {
		metric runrecord.ResourceMetric
		limit  uint64
	}{
		{runrecord.ResourceInputBytes, envelope.MaxInputBytes},
		{runrecord.ResourceOutputBytes, envelope.MaxOutputBytes},
		{runrecord.ResourcePeakHostBytes, envelope.HostBytes},
		{runrecord.ResourcePeakDeviceBytes, envelope.DeviceBytes},
	} {
		if observed, measured := resources.Measure(bound.metric); measured && observed > bound.limit {
			return errors.New("evaluation: capability probe exceeds its resource envelope")
		}
	}
	return nil
}

var productionProbeResourceMetrics = []runrecord.ResourceMetric{
	runrecord.ResourceInputTokens, runrecord.ResourceOutputTokens,
	runrecord.ResourceInputBytes, runrecord.ResourceOutputBytes,
	runrecord.ResourceWallNS, runrecord.ResourcePeakHostBytes,
	runrecord.ResourcePeakDeviceBytes, runrecord.ResourceHostToDeviceBytes,
	runrecord.ResourceDeviceToHostBytes,
}

func requireProbeResourceObservation(
	ctx context.Context,
	reader artifact.Reader,
	id, provider artifact.ID,
) (runrecord.ServingObservation, runrecord.ResourceFitness, error) {
	observation, err := runrecord.RequireServingAttemptObservation(ctx, reader, id)
	if err != nil {
		return runrecord.ServingObservation{}, runrecord.ResourceFitness{}, err
	}
	resources, err := observation.ResourceFitness(provider, nil, productionProbeResourceMetrics...)
	return observation, resources, err
}

func sameResourceFitness(left, right runrecord.ResourceFitness) bool {
	return left.Version == right.Version && left.Scope == right.Scope &&
		slices.Equal(left.Measures, right.Measures) && left.Interactions == nil && right.Interactions == nil
}

func productionEntry(
	profile ActivationProfile,
	selection modelrecipe.CapabilityEvidenceSelection,
	receipt runrecord.StageReceipt,
) (ProductionCapabilityEntry, error) {
	switch profile.Kind {
	case ActivationProfileLocalModel:
		if selection.Session != modelrecipe.SessionSpillover && receipt.Invocation == nil {
			return ProductionEntryLocalPlacement, nil
		}
	case ActivationProfilePeerModel:
		if selection.Session == modelrecipe.SessionSpillover && receipt.Invocation == nil {
			return ProductionEntryPeerPlacement, nil
		}
	case ActivationProfileAgentTool:
		if receipt.Invocation != nil && receipt.Invocation.Boundary == invocation.BoundaryAgent {
			return ProductionEntryAgentCoordinator, nil
		}
	}
	return "", errors.New("evaluation: no production entry matches the stored execution")
}

func stageReceiptOutcome(receipt runrecord.StageReceipt) (runrecord.Outcome, error) {
	switch receipt.State {
	case runrecord.StageCompleted:
		return runrecord.OutcomeSucceeded, nil
	case runrecord.StageFailed:
		return runrecord.OutcomeFailed, nil
	default:
		return "", errors.New("evaluation: capability probe receipt is not terminal")
	}
}

func probeDecisionMatchesOutcome(decision recipe.DecisionOutcome, outcome runrecord.Outcome) bool {
	return decision == recipe.DecisionAccepted && outcome == runrecord.OutcomeSucceeded ||
		decision == recipe.DecisionFailed && outcome == runrecord.OutcomeFailed
}

func validateAgentCoordinatorProbe(
	ctx context.Context,
	reader artifact.Reader,
	profileCapability runrecord.CapabilityIdentity,
	trace runrecord.InteractionTrace,
	receipt runrecord.StageReceipt,
) error {
	binding := receipt.Invocation
	if binding == nil || binding.Boundary != invocation.BoundaryAgent || binding.Kind != invocation.MutationTool {
		return errors.New("evaluation: agent capability probe lacks a coordinator invocation receipt")
	}
	manual, err := agenttool.RequireManual(ctx, reader, binding.Subject)
	if err != nil || manual.Capability != profileCapability.ID || manual.Name != binding.Action {
		return errors.Join(errors.New("evaluation: agent capability probe manual differs"), err)
	}
	stimulus, err := runrecord.RequireAttemptStimulus(ctx, reader, binding.Preflight)
	if err != nil || stimulus.Operation != receipt.Operation || stimulus.Manual != manual.ID ||
		stimulus.Arguments != binding.Arguments || stimulus.Effect != binding.Effect {
		return errors.Join(errors.New("evaluation: agent capability probe preflight differs"), err)
	}
	current, found, err := runrecord.ResolveAttemptStimulus(ctx, reader, stimulus.Operation, stimulus.Attempt)
	if err != nil || !found || current.ID != stimulus.ID {
		return errors.Join(errors.New("evaluation: agent capability probe preflight is stale"), err)
	}
	calls, failures, err := trace.ToolExchanges()
	if err != nil || failures != 0 || len(calls) != 1 {
		return errors.Join(errors.New("evaluation: agent capability probe lacks one completed tool exchange"), err)
	}
	call := calls[0]
	if call.Manual != manual.ID || call.Name != binding.Action {
		return errors.New("evaluation: agent capability probe tool exchange differs")
	}
	arguments, err := runrecord.AttemptArgumentContent([]byte(call.Arguments))
	if err != nil || arguments.Descriptor.ID != binding.Arguments {
		return errors.Join(errors.New("evaluation: agent capability probe arguments differ"), err)
	}
	var resultID artifact.ID
	for _, event := range trace.Events {
		if event.Kind == runrecord.InteractionEventToolResult && event.Message.ToolCallID == call.ID {
			resultID, err = artifact.IdentifyBytes(artifact.KindEvidence, []byte(event.Message.Content))
			break
		}
	}
	if err != nil || !resultID.Valid() || !stageBindingsContain(receipt.Outputs, resultID) ||
		!stageBindingsContain(receipt.Inputs, manual.ID) ||
		!stageBindingsContain(receipt.Inputs, binding.Arguments) ||
		!stageBindingsContain(receipt.Inputs, binding.Preflight) {
		return errors.Join(errors.New("evaluation: agent capability probe receipt payload differs"), err)
	}
	return nil
}

func requireCurrentStageReceipt(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (runrecord.StageReceipt, error) {
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil || !found {
		return runrecord.StageReceipt{}, errors.Join(errors.New("evaluation: capability probe stage receipt is absent"), err)
	}
	receipt, err := runrecord.ParseStageReceipt(content.Data)
	if err != nil || receipt.ID != id {
		return runrecord.StageReceipt{}, errors.Join(errors.New("evaluation: capability probe stage receipt differs"), err)
	}
	current, found, err := runrecord.ResolveStageReceipt(ctx, reader, receipt.Operation, receipt.Node)
	if err != nil || !found || current.ID != receipt.ID {
		return runrecord.StageReceipt{}, errors.Join(errors.New("evaluation: capability probe stage receipt is stale"), err)
	}
	return receipt, nil
}

func programContainsNode(selection modelrecipe.CapabilityEvidenceSelection, node recipe.NodeID) bool {
	for _, stage := range selection.Program.Stages() {
		if stage.Node.ID == node {
			return true
		}
	}
	return false
}

func stageReceiptMatchesOutcome(receipt runrecord.StageReceipt, outcome runrecord.Outcome) bool {
	return outcome == runrecord.OutcomeSucceeded && receipt.State == runrecord.StageCompleted ||
		outcome == runrecord.OutcomeFailed && receipt.State == runrecord.StageFailed
}

func stageBindingsContain(bindings []runrecord.StageBinding, id artifact.ID) bool {
	return slices.ContainsFunc(bindings, func(binding runrecord.StageBinding) bool {
		return slices.Contains(binding.Artifacts, id)
	})
}

func (entry ProductionCapabilityEntry) valid() bool {
	return entry == ProductionEntryLocalPlacement || entry == ProductionEntryPeerPlacement ||
		entry == ProductionEntryAgentCoordinator
}

func (entry ProductionCapabilityEntry) surface() runrecord.InteractionSurface {
	switch entry {
	case ProductionEntryLocalPlacement:
		return runrecord.SurfaceServing
	case ProductionEntryPeerPlacement:
		return runrecord.SurfacePeer
	case ProductionEntryAgentCoordinator:
		return runrecord.SurfaceAgent
	default:
		return ""
	}
}

func cloneCapabilityProbeResult(value CapabilityProbeResult) CapabilityProbeResult {
	value.Resources.Measures = slices.Clone(value.Resources.Measures)
	if value.Resources.Interactions != nil {
		work := *value.Resources.Interactions
		value.Resources.Interactions = &work
	}
	return value
}
