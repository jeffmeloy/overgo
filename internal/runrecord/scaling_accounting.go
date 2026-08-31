package runrecord

import (
	"errors"
	"math/big"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
)

const (
	scalingAccountingMediaType = "application/vnd.overgo.scaling-accounting+json"
	scalingAccountingSchema    = "overgo/scaling-accounting/v1"
)

type scalingCount struct {
	Decimal  string      `json:"decimal"`
	Method   artifact.ID `json:"method"`
	Evidence artifact.ID `json:"evidence"`
}

type scalingParameterAccounting struct {
	Inventory artifact.ID  `json:"inventory"`
	Total     scalingCount `json:"total"`
	Active    scalingCount `json:"active"`
}

type scalingTokenAccounting struct {
	Tokenizer artifact.ID  `json:"tokenizer"`
	Processed scalingCount `json:"processed"`
}

type scalingFLOPAccounting struct {
	Analytic scalingCount `json:"analytic"`
	Measured scalingCount `json:"measured"`
}

type scalingRuntimeAccounting struct {
	WallNS          scalingCount `json:"wall_ns"`
	PeakHostBytes   scalingCount `json:"peak_host_bytes"`
	PeakDeviceBytes scalingCount `json:"peak_device_bytes"`
}

type scalingRate struct {
	WorkDecimal   string `json:"work_decimal"`
	WallNSDecimal string `json:"wall_ns_decimal"`
}

type scalingThroughputAccounting struct {
	TokensPerWallNS        scalingRate `json:"tokens_per_wall_ns"`
	AnalyticFLOPsPerWallNS scalingRate `json:"analytic_flops_per_wall_ns"`
	MeasuredFLOPsPerWallNS scalingRate `json:"measured_flops_per_wall_ns"`
}

type scalingAccounting struct {
	ID          artifact.ID                 `json:"-"`
	Version     uint16                      `json:"version"`
	Run         artifact.ID                 `json:"run"`
	Model       artifact.ID                 `json:"model"`
	Definition  artifact.ID                 `json:"definition"`
	Dataset     artifact.ID                 `json:"dataset"`
	Split       artifact.ID                 `json:"split"`
	Recipe      artifact.ID                 `json:"recipe"`
	Checkpoint  artifact.ID                 `json:"checkpoint"`
	Seed        artifact.ID                 `json:"seed"`
	DataOrder   artifact.ID                 `json:"data_order"`
	Hardware    artifact.ID                 `json:"hardware"`
	Environment artifact.ID                 `json:"environment"`
	Code        artifact.ID                 `json:"code"`
	Parameters  scalingParameterAccounting  `json:"parameters"`
	Tokens      scalingTokenAccounting      `json:"tokens"`
	FLOPs       scalingFLOPAccounting       `json:"flops"`
	Runtime     scalingRuntimeAccounting    `json:"runtime"`
	Throughput  scalingThroughputAccounting `json:"throughput"`
}

var scalingAccountingCodec = artifact.JSONDocumentCodec(
	"scaling accounting", artifact.KindEvidence, scalingAccountingMediaType, scalingAccountingSchema,
	canonicalizeScalingAccounting,
	func(value scalingAccounting) artifact.ID { return value.ID },
	func(value *scalingAccounting, id artifact.ID) { value.ID = id }, nil,
)

var scalingAccountingLineage = func(value scalingAccounting) []artifact.Lineage {
	counts := [...]scalingCount{
		value.Parameters.Total, value.Parameters.Active, value.Tokens.Processed,
		value.FLOPs.Analytic, value.FLOPs.Measured,
		value.Runtime.WallNS, value.Runtime.PeakHostBytes, value.Runtime.PeakDeviceBytes,
	}
	parents := []artifact.ID{
		value.Run, value.Model, value.Definition, value.Dataset, value.Split, value.Recipe, value.Checkpoint,
		value.Seed, value.DataOrder, value.Hardware, value.Environment, value.Code,
		value.Parameters.Inventory, value.Tokens.Tokenizer,
	}
	for _, count := range counts {
		parents = append(parents, count.Method, count.Evidence)
	}
	return artifact.DependencyLineage(value.ID, uniqueScalingAuthorities(parents)...)
}

func canonicalizeScalingAccounting(value *scalingAccounting) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Run.Kind() != artifact.KindRun ||
		value.Model.Kind() != artifact.KindModel || value.Definition.Kind() != artifact.KindModelDefinition ||
		value.Dataset.Kind() != artifact.KindDataset || value.Split.Kind() != artifact.KindDatasetShard ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Checkpoint.Kind() != artifact.KindCheckpoint ||
		value.Seed.Kind() != artifact.KindEvidence || value.DataOrder.Kind() != artifact.KindEvidence ||
		value.Hardware.Kind() != artifact.KindEvidence || value.Environment.Kind() != artifact.KindEvidence ||
		value.Code.Kind() != artifact.KindEvidence || value.Parameters.Inventory.Kind() != artifact.KindTensorInventory ||
		value.Tokens.Tokenizer.Kind() != artifact.KindTokenizer {
		return errors.New("run record: invalid scaling accounting authorities")
	}
	counts := [...]scalingCount{
		value.Parameters.Total, value.Parameters.Active, value.Tokens.Processed,
		value.FLOPs.Analytic, value.FLOPs.Measured,
		value.Runtime.WallNS, value.Runtime.PeakHostBytes, value.Runtime.PeakDeviceBytes,
	}
	parsed := make([]*big.Int, len(counts))
	for index, count := range counts {
		if count.Method.Kind() != artifact.KindProfile || count.Evidence.Kind() != artifact.KindEvidence {
			return errors.New("run record: scaling count lacks method or evidence authority")
		}
		parsed[index] = ParseScalingDecimal(count.Decimal)
		if parsed[index] == nil {
			return errors.New("run record: scaling count is not a canonical nonnegative integer")
		}
	}
	if parsed[0].Sign() == 0 || parsed[1].Sign() == 0 || parsed[2].Sign() == 0 || parsed[5].Sign() == 0 ||
		parsed[1].Cmp(parsed[0]) > 0 || value.FLOPs.Analytic.Method == value.FLOPs.Measured.Method {
		return errors.New("run record: scaling work is empty, inactive, or conflates analytic and measured FLOPs")
	}
	wall := value.Runtime.WallNS.Decimal
	if value.Throughput.TokensPerWallNS != (scalingRate{WorkDecimal: value.Tokens.Processed.Decimal, WallNSDecimal: wall}) ||
		value.Throughput.AnalyticFLOPsPerWallNS != (scalingRate{WorkDecimal: value.FLOPs.Analytic.Decimal, WallNSDecimal: wall}) ||
		value.Throughput.MeasuredFLOPsPerWallNS != (scalingRate{WorkDecimal: value.FLOPs.Measured.Decimal, WallNSDecimal: wall}) {
		return errors.New("run record: scaling throughput differs from exact work and wall time")
	}
	return nil
}

// ParseScalingDecimal returns an arbitrary-precision canonical nonnegative decimal, or nil for a noncanonical value.
func ParseScalingDecimal(value string) *big.Int {
	if value == "" || len(value) > 1 && value[0] == '0' {
		return nil
	}
	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return nil
		}
	}
	parsed, ok := new(big.Int).SetString(value, binaryschema.DecimalRadix)
	if !ok {
		return nil
	}
	return parsed
}

func uniqueScalingAuthorities(ids []artifact.ID) []artifact.ID {
	result := make([]artifact.ID, 0, len(ids))
	seen := make(map[artifact.ID]bool, len(ids))
	for _, id := range ids {
		if !seen[id] {
			seen[id] = true
			result = append(result, id)
		}
	}
	return result
}
