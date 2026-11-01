package modelartifact

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

// PreserveRawFileDescriptors keeps compatible immutable facts for source files.
// Generic binary declarations inherit existing schema-free media facts.
// Plain text, Markdown and Python declarations retain their source interpretations.
// Conflicting sizes, schemas and other specific media declarations are refused.
func PreserveRawFileDescriptors(ctx context.Context, reader artifact.Reader, batch *artifact.Batch) error {
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
		rawText := func(media string) bool {
			return media == "text/plain" || media == "text/markdown" || media == "text/x-python"
		}
		generic := declared.MediaType == binaryMediaType && declared.Schema == ""
		if !generic && !rawText(declared.MediaType) {
			return nil
		}
		existing, found, err := reader.Artifact(ctx, declared.ID)
		if err != nil {
			return err
		}
		if !found || existing == *declared {
			return nil
		}
		textViews := rawText(existing.MediaType) && rawText(declared.MediaType)
		if existing.Size != declared.Size || existing.Schema != "" || declared.Schema != "" || (!textViews && !generic) {
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
