package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestScalingAccountingBindsTotalActiveMeasuredAndAnalyticCosts(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	count := func(name, decimal string) scalingCount {
		return scalingCount{
			Decimal: decimal, Method: id(artifact.KindProfile, name+" method"), Evidence: id(artifact.KindEvidence, name+" evidence"),
		}
	}
	value := scalingAccounting{
		Version: artifact.InitialDocumentVersion,
		Run:     id(artifact.KindRun, "run"), Model: id(artifact.KindModel, "model"),
		Definition: id(artifact.KindModelDefinition, "definition"), Dataset: id(artifact.KindDataset, "dataset"),
		Split: id(artifact.KindDatasetShard, "split"), Recipe: id(artifact.KindRecipe, "recipe"),
		Checkpoint: id(artifact.KindCheckpoint, "checkpoint"), Seed: id(artifact.KindEvidence, "seed"),
		DataOrder: id(artifact.KindEvidence, "data order"), Hardware: id(artifact.KindEvidence, "hardware"),
		Environment: id(artifact.KindEvidence, "environment"), Code: id(artifact.KindEvidence, "code"),
		Parameters: scalingParameterAccounting{
			Inventory: id(artifact.KindTensorInventory, "inventory"),
			Total:     count("total parameters", "70000000000"), Active: count("active parameters", "12000000000"),
		},
		Tokens: scalingTokenAccounting{
			Tokenizer: id(artifact.KindTokenizer, "tokenizer"), Processed: count("processed tokens", "15000000000000"),
		},
		FLOPs: scalingFLOPAccounting{
			Analytic: count("analytic flops", "6300000000000000000000000"),
			Measured: count("measured flops", "5985000000000000000000000"),
		},
		Runtime: scalingRuntimeAccounting{
			WallNS: count("wall time", "2592000000000000"), PeakHostBytes: count("peak host memory", "549755813888"),
			PeakDeviceBytes: count("peak device memory", "85899345920"),
		},
	}
	value.Throughput = scalingThroughputAccounting{
		TokensPerWallNS:        scalingRate{WorkDecimal: value.Tokens.Processed.Decimal, WallNSDecimal: value.Runtime.WallNS.Decimal},
		AnalyticFLOPsPerWallNS: scalingRate{WorkDecimal: value.FLOPs.Analytic.Decimal, WallNSDecimal: value.Runtime.WallNS.Decimal},
		MeasuredFLOPsPerWallNS: scalingRate{WorkDecimal: value.FLOPs.Measured.Decimal, WallNSDecimal: value.Runtime.WallNS.Decimal},
	}
	accounting, err := scalingAccountingCodec.New(value)
	if err != nil {
		t.Fatal(err)
	}
	content, err := scalingAccountingCodec.Content(accounting)
	if err != nil || content.Descriptor.ID != accounting.ID || len(scalingAccountingLineage(accounting)) < 20 {
		t.Fatalf("scaling accounting = (%+v, %v)", accounting, err)
	}
	activeExceedsTotal := value
	activeExceedsTotal.Parameters.Active.Decimal = "70000000001"
	if _, err := scalingAccountingCodec.New(activeExceedsTotal); err == nil {
		t.Fatal("active parameters exceeded total parameters")
	}
	missingMeasuredAuthority := value
	missingMeasuredAuthority.FLOPs.Measured.Evidence = artifact.ID{}
	if _, err := scalingAccountingCodec.New(missingMeasuredAuthority); err == nil {
		t.Fatal("measured FLOPs lacked evidence")
	}
	unitMismatch := value
	unitMismatch.Throughput.MeasuredFLOPsPerWallNS.WallNSDecimal = value.Tokens.Processed.Decimal
	if _, err := scalingAccountingCodec.New(unitMismatch); err == nil {
		t.Fatal("throughput used a non-time denominator")
	}
	conflatedMethods := value
	conflatedMethods.FLOPs.Measured.Method = value.FLOPs.Analytic.Method
	if _, err := scalingAccountingCodec.New(conflatedMethods); err == nil {
		t.Fatal("analytic and measured FLOPs shared one method authority")
	}
}
