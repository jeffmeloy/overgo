package composition

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/tensor"
)

const (
	offlineTensorResourcePolicyVersion uint16 = artifact.InitialDocumentVersion
	offlineTensorExecutionPlanVersion  uint16 = artifact.InitialDocumentVersion
)

const (
	offlineTensorResourcePolicyMediaType = "application/vnd.overgo.offline-tensor-resource-policy+json"
	offlineTensorResourcePolicySchema    = "overgo/offline-tensor-resource-policy/v1"
	offlineTensorExecutionPlanMediaType  = "application/vnd.overgo.offline-tensor-execution-plan+json"
	offlineTensorExecutionPlanSchema     = "overgo/offline-tensor-execution-plan/v1"
)

// OfflineTensorResourcePolicy is the recorded resource decision consumed by
// offline artifact planning. Numeric limits have no implicit defaults.
type OfflineTensorResourcePolicy struct {
	Version          uint16           `json:"version"`
	Placement        recipe.Placement `json:"placement"`
	MaxResidentBytes uint64           `json:"max_resident_bytes"`
	MaxShardBytes    uint64           `json:"max_shard_bytes"`
	Rationale        string           `json:"rationale"`
	ReopenTrigger    string           `json:"reopen_trigger"`
	ID               artifact.ID      `json:"-"`
}

// OfflineTensorLifetime identifies the inclusive operation interval retaining
// one materialized tensor result.
type OfflineTensorLifetime struct {
	First uint32 `json:"first"`
	Last  uint32 `json:"last"`
}

// OfflineTensorOperation is one exact tensor transformation in execution order.
type OfflineTensorOperation struct {
	Index         uint32                `json:"index"`
	Name          string                `json:"name"`
	Storage       string                `json:"storage"`
	SourceBytes   []uint64              `json:"source_bytes"`
	OutputBytes   uint64                `json:"output_bytes"`
	ResidentBytes uint64                `json:"resident_bytes"`
	Placement     recipe.Placement      `json:"placement"`
	Lifetime      OfflineTensorLifetime `json:"lifetime"`
	Shard         uint32                `json:"shard"`
}

// OfflineTensorShard is a deterministic consecutive output partition.
type OfflineTensorShard struct {
	Index   uint32   `json:"index"`
	Bytes   uint64   `json:"bytes"`
	Tensors []string `json:"tensors"`
}

// OfflineTensorExecutionPlan binds exact artifact authority to derived tensor
// operations and their bounded residency and output partitions.
type OfflineTensorExecutionPlan struct {
	Version           uint16                   `json:"version"`
	ArtifactPlan      artifact.ID              `json:"artifact_plan"`
	ResourcePolicy    artifact.ID              `json:"resource_policy"`
	Operator          OfflineArtifactOperator  `json:"operator"`
	Inputs            []OfflineArtifactModel   `json:"inputs"`
	Operations        []OfflineTensorOperation `json:"operations"`
	Shards            []OfflineTensorShard     `json:"shards"`
	PeakResidentBytes uint64                   `json:"peak_resident_bytes"`
	ID                artifact.ID              `json:"-"`
}

var offlineTensorResourcePolicyCodec = artifact.JSONDocumentCodec(
	"offline tensor resource policy", artifact.KindProfile,
	offlineTensorResourcePolicyMediaType, offlineTensorResourcePolicySchema,
	canonicalizeOfflineTensorResourcePolicy,
	func(value OfflineTensorResourcePolicy) artifact.ID { return value.ID },
	func(value *OfflineTensorResourcePolicy, id artifact.ID) { value.ID = id },
	func(value OfflineTensorResourcePolicy) OfflineTensorResourcePolicy { return value },
)

var offlineTensorExecutionPlanCodec = artifact.JSONDocumentCodec(
	"offline tensor execution plan", artifact.KindProfile,
	offlineTensorExecutionPlanMediaType, offlineTensorExecutionPlanSchema,
	canonicalizeOfflineTensorExecutionPlan,
	func(value OfflineTensorExecutionPlan) artifact.ID { return value.ID },
	func(value *OfflineTensorExecutionPlan, id artifact.ID) { value.ID = id },
	cloneOfflineTensorExecutionPlan,
)

// NewOfflineTensorResourcePolicy records explicit resource limits and their
// decision rationale. The compiler never supplies fallback values.
func NewOfflineTensorResourcePolicy(
	placement recipe.Placement,
	maxResidentBytes, maxShardBytes uint64,
	rationale, reopenTrigger string,
) (OfflineTensorResourcePolicy, error) {
	return offlineTensorResourcePolicyCodec.New(OfflineTensorResourcePolicy{
		Version: offlineTensorResourcePolicyVersion, Placement: placement,
		MaxResidentBytes: maxResidentBytes, MaxShardBytes: maxShardBytes,
		Rationale: rationale, ReopenTrigger: reopenTrigger,
	})
}

// Content returns the immutable resource-policy document.
func (value OfflineTensorResourcePolicy) Content() (artifact.Content, error) {
	return offlineTensorResourcePolicyCodec.Content(value)
}

// ValidateIdentity verifies the resource-policy content identity.
func (value OfflineTensorResourcePolicy) ValidateIdentity() error {
	return offlineTensorResourcePolicyCodec.ValidateIdentity(value)
}

// Batch prepares publication of the resource-policy document.
func (value OfflineTensorResourcePolicy) Batch(key string) (artifact.Batch, error) {
	return offlineTensorResourcePolicyCodec.Batch(key, value, nil, nil)
}

// CompileOfflineTensorExecutionPlan resolves the immutable recipe and resource
// policy and derives every tensor operation from exact input inventories.
func CompileOfflineTensorExecutionPlan(
	ctx context.Context,
	reader artifact.Reader,
	artifactPlan OfflineArtifactPlan,
	policy OfflineTensorResourcePolicy,
) (OfflineTensorExecutionPlan, error) {
	if ctx == nil || reader == nil {
		return OfflineTensorExecutionPlan{}, errors.New("composition: offline tensor authorities are absent")
	}
	if err := offlineArtifactPlanCodec.ValidateIdentity(artifactPlan); err != nil {
		return OfflineTensorExecutionPlan{}, fmt.Errorf("composition: invalid offline artifact plan authority: %w", err)
	}
	if err := policy.ValidateIdentity(); err != nil {
		return OfflineTensorExecutionPlan{}, fmt.Errorf("composition: invalid offline tensor resource policy: %w", err)
	}
	if policy.Placement != recipe.PlacementHost {
		return OfflineTensorExecutionPlan{}, errors.New("composition: offline tensor placement is unsupported")
	}

	inventories := make([]modelartifact.TensorInventoryDocument, len(artifactPlan.Inputs))
	for index, input := range artifactPlan.Inputs {
		inventory, found, loadErr := modelartifact.ReadTensorInventoryDocument(ctx, reader, input.Inventory)
		if loadErr != nil || !found || inventory.Owner != input.Model || inventory.ID != input.Inventory {
			return OfflineTensorExecutionPlan{}, errors.Join(
				fmt.Errorf("composition: offline tensor input %d inventory is absent", index), loadErr,
			)
		}
		inventories[index] = inventory
	}
	operations, shards, peak, err := deriveOfflineTensorOperations(artifactPlan, policy, inventories)
	if err != nil {
		return OfflineTensorExecutionPlan{}, err
	}
	return offlineTensorExecutionPlanCodec.New(OfflineTensorExecutionPlan{
		Version: offlineTensorExecutionPlanVersion, ArtifactPlan: artifactPlan.ID,
		ResourcePolicy: policy.ID, Operator: artifactPlan.Operator,
		Inputs: artifactPlan.Inputs, Operations: operations, Shards: shards,
		PeakResidentBytes: peak,
	})
}

// Content returns the immutable tensor-execution plan document.
func (value OfflineTensorExecutionPlan) Content() (artifact.Content, error) {
	return offlineTensorExecutionPlanCodec.Content(value)
}

// ValidateIdentity verifies the complete execution plan and its content
// identity before an offline executor consumes it.
func (value OfflineTensorExecutionPlan) ValidateIdentity() error {
	return offlineTensorExecutionPlanCodec.ValidateIdentity(value)
}

// LoadOfflineTensorExecutionPlan requires one exact immutable execution plan.
func LoadOfflineTensorExecutionPlan(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (OfflineTensorExecutionPlan, error) {
	return offlineTensorExecutionPlanCodec.Require(ctx, reader, id)
}

// Lineage binds the tensor-execution plan to every exact input authority.
func (value OfflineTensorExecutionPlan) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.ArtifactPlan, value.ResourcePolicy}
	for _, input := range value.Inputs {
		parents = append(parents, input.Model, input.Definition, input.Profile, input.Inventory)
	}
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

// Batch prepares publication of the tensor-execution plan and its lineage.
func (value OfflineTensorExecutionPlan) Batch(key string) (artifact.Batch, error) {
	return offlineTensorExecutionPlanCodec.Batch(key, value, value.Lineage(), nil)
}

func deriveOfflineTensorOperations(
	plan OfflineArtifactPlan,
	policy OfflineTensorResourcePolicy,
	inventories []modelartifact.TensorInventoryDocument,
) ([]OfflineTensorOperation, []OfflineTensorShard, uint64, error) {
	reference := inventories[tensor.FirstOffset]
	operations := make([]OfflineTensorOperation, len(reference.Tensors))
	shards := make([]OfflineTensorShard, 0, len(reference.Tensors))
	var peak uint64
	for index, fact := range reference.Tensors {
		outputBytes, residentBytes, err := offlineTensorExtents(plan.Operator, fact, len(inventories))
		if err != nil {
			return nil, nil, 0, fmt.Errorf("composition: offline tensor %q: %w", fact.Name, err)
		}
		if residentBytes > policy.MaxResidentBytes {
			return nil, nil, 0, fmt.Errorf("composition: offline tensor %q exceeds resident policy", fact.Name)
		}
		if outputBytes > policy.MaxShardBytes {
			return nil, nil, 0, fmt.Errorf("composition: offline tensor %q exceeds shard policy", fact.Name)
		}
		peak = max(peak, residentBytes)
		shard := len(shards)
		if shard == 0 || shards[shard-tensor.SingletonExtent].Bytes > policy.MaxShardBytes-outputBytes {
			shards = append(shards, OfflineTensorShard{Index: uint32(shard)})
		} else {
			shard--
		}
		current := &shards[shard]
		current.Bytes += outputBytes
		current.Tensors = append(current.Tensors, fact.Name)
		sourceBytes := make([]uint64, len(inventories))
		for source := range inventories {
			candidate := inventories[source].Tensors[index]
			if candidate.Name != fact.Name || candidate.Storage != fact.Storage ||
				candidate.Bytes != fact.Bytes || !slices.Equal(candidate.Shape, fact.Shape) {
				return nil, nil, 0, fmt.Errorf("composition: offline tensor %q inventory differs", fact.Name)
			}
			sourceBytes[source] = candidate.Bytes
		}
		position := uint32(index)
		operations[index] = OfflineTensorOperation{
			Index: position, Name: fact.Name, Storage: fact.Storage,
			SourceBytes: sourceBytes, OutputBytes: outputBytes, ResidentBytes: residentBytes,
			Placement: policy.Placement, Lifetime: OfflineTensorLifetime{First: position, Last: position},
			Shard: current.Index,
		}
	}
	return operations, shards, peak, nil
}

func offlineTensorExtents(operator OfflineArtifactOperator, fact modelartifact.TensorFact, inputs int) (uint64, uint64, error) {
	switch operator {
	case OfflineArtifactExactPassthrough:
		return fact.Bytes, fact.Bytes, nil
	case OfflineArtifactTaskArithmetic:
		if inputs < tensor.PairedExtent || fact.Storage != "f32" {
			return 0, 0, errors.New("task arithmetic requires paired F32 inputs")
		}
		residentBytes, ok := checked.Mul64(fact.Bytes, uint64(tensor.PairedExtent))
		if !ok {
			return 0, 0, errors.New("resident extent overflows")
		}
		return fact.Bytes, residentBytes, nil
	default:
		return 0, 0, errors.New("unsupported offline tensor operator")
	}
}

func canonicalizeOfflineTensorResourcePolicy(value *OfflineTensorResourcePolicy) error {
	if value == nil || value.Version != offlineTensorResourcePolicyVersion ||
		value.Placement != recipe.PlacementHost || !checked.Nonzero(value.MaxResidentBytes) ||
		!checked.Nonzero(value.MaxShardBytes) || strings.TrimSpace(value.Rationale) != value.Rationale ||
		strings.TrimSpace(value.ReopenTrigger) != value.ReopenTrigger || value.Rationale == "" || value.ReopenTrigger == "" ||
		strings.ContainsAny(value.Rationale, "\x00\r\n") || strings.ContainsAny(value.ReopenTrigger, "\x00\r\n") {
		return errors.New("composition: invalid offline tensor resource policy")
	}
	return nil
}

func canonicalizeOfflineTensorExecutionPlan(value *OfflineTensorExecutionPlan) error {
	if value == nil || value.Version != offlineTensorExecutionPlanVersion ||
		value.ArtifactPlan.Kind() != artifact.KindRecipe || value.ResourcePolicy.Kind() != artifact.KindProfile ||
		(value.Operator != OfflineArtifactExactPassthrough && value.Operator != OfflineArtifactTaskArithmetic) ||
		len(value.Inputs) == 0 || len(value.Operations) == 0 || len(value.Shards) == 0 ||
		!checked.Nonzero(value.PeakResidentBytes) {
		return errors.New("composition: invalid offline tensor execution plan")
	}
	if value.Operator == OfflineArtifactExactPassthrough && len(value.Inputs) != tensor.SingletonExtent ||
		value.Operator == OfflineArtifactTaskArithmetic && len(value.Inputs) < tensor.PairedExtent {
		return errors.New("composition: offline tensor execution input count differs")
	}
	var peak uint64
	shardBytes := make([]uint64, len(value.Shards))
	shardNames := make([][]string, len(value.Shards))
	for index, operation := range value.Operations {
		position := uint32(index)
		if operation.Index != position || operation.Name == "" || operation.Placement != recipe.PlacementHost ||
			operation.Storage == "" || len(operation.SourceBytes) != len(value.Inputs) ||
			!checked.Nonzero(operation.OutputBytes) || !allNonzero(operation.SourceBytes) ||
			!checked.Nonzero(operation.ResidentBytes) || operation.ResidentBytes > value.PeakResidentBytes ||
			operation.Lifetime != (OfflineTensorLifetime{First: position, Last: position}) ||
			int(operation.Shard) >= len(value.Shards) {
			return errors.New("composition: invalid offline tensor operation")
		}
		if index > tensor.FirstOffset && value.Operations[index-tensor.SingletonExtent].Name >= operation.Name {
			return errors.New("composition: offline tensor operations are unordered")
		}
		if operation.ResidentBytes > peak {
			peak = operation.ResidentBytes
		}
		shard := int(operation.Shard)
		var ok bool
		shardBytes[shard], ok = checked.Add64(shardBytes[shard], operation.OutputBytes)
		if !ok {
			return errors.New("composition: offline tensor shard extent overflows")
		}
		shardNames[shard] = append(shardNames[shard], operation.Name)
	}
	if peak != value.PeakResidentBytes {
		return errors.New("composition: offline tensor peak residency differs")
	}
	for index, shard := range value.Shards {
		if shard.Index != uint32(index) || !checked.Nonzero(shard.Bytes) ||
			shard.Bytes != shardBytes[index] || !slices.Equal(shard.Tensors, shardNames[index]) {
			return errors.New("composition: invalid offline tensor shard")
		}
	}
	return nil
}

func allNonzero(values []uint64) bool {
	for _, value := range values {
		if !checked.Nonzero(value) {
			return false
		}
	}
	return true
}

func cloneOfflineTensorExecutionPlan(value OfflineTensorExecutionPlan) OfflineTensorExecutionPlan {
	value.Inputs = slices.Clone(value.Inputs)
	value.Operations = slices.Clone(value.Operations)
	for index := range value.Operations {
		value.Operations[index].SourceBytes = slices.Clone(value.Operations[index].SourceBytes)
	}
	value.Shards = slices.Clone(value.Shards)
	for index := range value.Shards {
		value.Shards[index].Tensors = slices.Clone(value.Shards[index].Tensors)
	}
	return value
}
