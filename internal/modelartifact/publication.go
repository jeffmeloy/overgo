package modelartifact

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

// PreserveRawTextDescriptors keeps immutable catalog facts when two publishers
// describe the same raw text bytes as plain text and Markdown. Their source
// declarations retain each interpretation. Sizes, schemas, tensor containers
// and every other media-type conflict remain strict.
func PreserveRawTextDescriptors(ctx context.Context, reader artifact.Reader, batch *artifact.Batch) error {
	if ctx == nil || reader == nil || batch == nil {
		return errors.New("model artifact: nil text publication input")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	preserve := func(declared *artifact.Descriptor) error {
		if declared.ID.Kind() != artifact.KindFile {
			return nil
		}
		existing, found, err := reader.Artifact(ctx, declared.ID)
		if err != nil {
			return err
		}
		if !found || existing == *declared {
			return nil
		}
		textViews := existing.MediaType == "text/plain" && declared.MediaType == "text/markdown" ||
			existing.MediaType == "text/markdown" && declared.MediaType == "text/plain"
		if existing.Size != declared.Size || existing.Schema != "" || declared.Schema != "" || !textViews {
			return fmt.Errorf("model artifact: conflicting raw-file facts for %s", declared.ID)
		}
		*declared = existing
		return nil
	}
	for index := range batch.Artifacts {
		if err := preserve(&batch.Artifacts[index]); err != nil {
			return err
		}
	}
	for index := range batch.Contents {
		if err := preserve(&batch.Contents[index].Descriptor); err != nil {
			return err
		}
	}
	return nil
}
