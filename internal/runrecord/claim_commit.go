package runrecord

import (
	"context"
	"errors"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
)

// CommitVerificationClaim lands one verification record with everything
// its claims stand on: the evidence document commits by content when
// the store does not already hold it, and the weights file registers
// with its on-disk location when the model is new to the store. Every
// claim path -- the compatibility CLI, the benchmark publisher --
// shares this landing so a claim can never reference evidence or a
// model the store cannot produce.
func CommitVerificationClaim(
	ctx context.Context,
	store artifact.Repository,
	record ModelVerification,
	evidenceData []byte,
	evidence artifact.ID,
	modelFile string,
) error {
	if len(record.Claims) == 0 {
		return errors.New("run record: verification record carries no claims")
	}
	batch, err := record.Batch("verification/" + record.ID.String())
	if err != nil {
		return err
	}
	if _, ok, err := artifact.ReadContent(ctx, store, evidence); err != nil {
		return err
	} else if !ok {
		batch.Contents = append(batch.Contents, artifact.Content{
			Descriptor: artifact.Descriptor{ID: evidence, Size: uint64(len(evidenceData))}, Data: evidenceData,
		})
	}
	if _, ok, err := store.Artifact(ctx, record.Model); err != nil {
		return err
	} else if !ok {
		absolute, err := filepath.Abs(modelFile)
		if err != nil {
			return err
		}
		info, err := os.Stat(modelFile)
		if err != nil {
			return err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: record.Model, Size: uint64(info.Size())})
		batch.Locations = append(batch.Locations, artifact.LocationEvent{
			Location: artifact.Location{Artifact: record.Model, Kind: artifact.LocationFile, Value: absolute},
			Action:   artifact.LocationAdd,
		})
	}
	_, err = store.Commit(ctx, batch)
	return err
}
