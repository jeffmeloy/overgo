// Command compatibility renders the model report, publishes the media
// report, exports media samples, publishes and checks the capability census,
// and commits verification records. The claims and the matrices published
// from them are cmd/validate's.
package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.Main(run)
}

func run() error {
	return runArgs(os.Args[1:])
}

// runArgs runs the command over its arguments with a flag set of its own,
// so a test drives it without touching the process's flags or arguments.
func runArgs(args []string) error {
	flags := flag.NewFlagSet("compatibility", flag.ExitOnError)
	models := flags.Bool("models", false, "print the nested model compatibility report joined against the store")
	updateMedia := flags.Bool("update-media", false, "write JSON measurements and the media capability report from store activations and verification claims")
	mediaScopeName := flags.String("media-scope", "all", "sample export tasks: all or image-video; the report always includes all media")
	publishCensus := flags.Bool("publish-census", false, "publish the complete registered capability denominator from clean code")
	checkCensus := flags.String("check-census", "", "check one exact immutable capability census against the live registered denominator")
	exportSamples := flags.Bool("export-samples", false, "export healthy media activations' generation outputs as decodable files under docs/media_samples")
	modelsRepo := flags.String("models-repo", "overgodb-store", "OvergoDB store for -models prototype resolution")
	recordVerification := flags.String("record-verification", "", "commit a typed model-verification record from a JSON spec (model, name, evidenced capability claims)")
	recordStore := flags.String("record", "", "OvergoDB root for -record-verification")
	if err := flags.Parse(args); err != nil {
		return err
	}
	mediaScope, err := resolveMediaReportScope(*mediaScopeName)
	if err != nil {
		return err
	}
	if *mediaScopeName != "all" && (!*exportSamples || *updateMedia) || *updateMedia && *exportSamples {
		return errors.New("usage: compatibility -update-media [-models-repo <overgodb>] or -export-samples [-media-scope all|image-video] [-models-repo <overgodb>]; -update-media always writes the complete docs/MEDIA_REPORT.md")
	}
	if *publishCensus || *checkCensus != "" {
		if flags.NArg() != 0 || *publishCensus && *checkCensus != "" || *models || *updateMedia || *exportSamples || *recordVerification != "" {
			return errors.New("usage: compatibility [-publish-census|-check-census <id>] [-models-repo <overgodb>]")
		}
		return runCapabilityCensus(*modelsRepo, *publishCensus, *checkCensus, os.Stdout)
	}
	if *exportSamples {
		if flags.NArg() != 0 || *recordVerification != "" {
			return errors.New("usage: compatibility -export-samples [-models-repo <overgodb>]")
		}
		return exportMediaSamples(".", *modelsRepo, mediaScope)
	}
	if *updateMedia {
		if flags.NArg() != 0 || *recordVerification != "" {
			return errors.New("usage: compatibility -update-media [-models-repo <overgodb>]")
		}
		data, err := generateMediaReport(".", *modelsRepo)
		if err != nil {
			return err
		}
		if err := clioptions.WriteOutputFile(filepath.FromSlash(mediaProjectionPath), data.JSON); err != nil {
			return err
		}
		if err := clioptions.WriteOutputFile(filepath.FromSlash(mediaReportPath), data.Markdown); err != nil {
			return err
		}
		fmt.Printf("wrote %s and %s\n", mediaProjectionPath, mediaReportPath)
		return nil
	}
	if *models {
		if flags.NArg() != 0 || *recordVerification != "" {
			return errors.New("usage: compatibility -models [-models-repo <overgodb>]")
		}
		store, err := overgodb.OpenReadOnly(*modelsRepo)
		if err != nil {
			return err
		}
		data, err := generateModelsReport(".", store)
		if err == nil {
			_, err = os.Stdout.Write(data)
		}
		return errors.Join(err, store.Close())
	}
	if *recordVerification != "" {
		if flags.NArg() != 0 || *recordStore == "" {
			return errors.New("usage: compatibility -record-verification <spec.json> -record <overgodb>")
		}
		return runRecordVerification(*recordVerification, *recordStore, os.Stdout)
	}
	return errors.New("usage: compatibility -models|-update-media|-export-samples|-publish-census|-check-census <id>|-record-verification <spec.json>; the claims and the matrices published from them are checked by go run ./cmd/validate -claims")
}

// runRecordVerification commits one typed model-verification record: the
// claims discipline as a store artifact rather than a document -- every
// capability tier grounded in named evidence, with lineage to all of it.
// The spec may ground its references: evidence files commit by content
// (their identity must match a claimed evidence ID) and the model file
// registers by descriptor and location -- both verified against the named
// identities, never invented.
func runRecordVerification(specPath, recordStore string, output io.Writer) error {
	var specification verificationSpecification
	if err := jsonfile.Decode(specPath, &specification); err != nil {
		return err
	}
	record, err := runrecord.NewModelVerificationCorrection(
		specification.Model, specification.Name, specification.Claims, specification.Supersedes,
	)
	if err != nil {
		return err
	}
	claimed := map[artifact.ID]bool{}
	for _, claim := range record.Claims {
		for _, evidence := range claim.Evidence {
			claimed[evidence] = true
		}
	}
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	for _, superseded := range record.Supersedes {
		content, ok, err := artifact.ReadContent(ctx, store, superseded)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("superseded model verification %s is absent", superseded)
		}
		prior, err := runrecord.ParseModelVerification(content.Data)
		if err != nil {
			return fmt.Errorf("parse superseded model verification %s: %w", superseded, err)
		}
		if prior.Model != record.Model {
			return fmt.Errorf("superseded model verification %s names model %s, correction names %s", superseded, prior.Model, record.Model)
		}
	}
	batch, err := record.Batch("verification/" + record.ID.String())
	if err != nil {
		return err
	}
	datasets := map[artifact.ID]bool{}
	for _, claim := range record.Claims {
		if claim.Dataset.Valid() {
			datasets[claim.Dataset] = true
		}
	}
	ground := func(paths []string, kind artifact.Kind, referenced map[artifact.ID]bool, role string) error {
		for _, path := range paths {
			data, err := os.ReadFile(path)
			if err != nil {
				return fmt.Errorf("%s file %s: %w", role, path, err)
			}
			identity, _, err := artifact.Identify(kind, bytes.NewReader(data))
			if err != nil {
				return err
			}
			if !referenced[identity] {
				return fmt.Errorf("%s file %s identifies as %s, which no claim references", role, path, identity)
			}
			if _, ok, err := artifact.ReadContent(ctx, store, identity); err != nil {
				return err
			} else if !ok {
				batch.Contents = append(batch.Contents, artifact.Content{
					Descriptor: artifact.Descriptor{ID: identity, Size: uint64(len(data))}, Data: data,
				})
			}
		}
		return nil
	}
	if err := ground(specification.EvidenceFiles, artifact.KindEvidence, claimed, "evidence"); err != nil {
		return err
	}
	if err := ground(specification.DatasetFiles, artifact.KindDatasetShard, datasets, "dataset"); err != nil {
		return err
	}
	if _, ok, err := store.Artifact(ctx, record.Model); err != nil {
		return err
	} else if !ok {
		if specification.ModelFile == "" {
			return fmt.Errorf("model %s is not in the store; the spec must name model_file to register it", record.Model)
		}
		file, err := os.Open(specification.ModelFile)
		if err != nil {
			return err
		}
		identity, size, err := artifact.Identify(artifact.KindModel, file)
		closeErr := file.Close()
		if err != nil || closeErr != nil {
			return errors.Join(err, closeErr)
		}
		if identity != record.Model {
			return fmt.Errorf("model file %s identifies as %s, spec claims %s", specification.ModelFile, identity, record.Model)
		}
		absolute, err := filepath.Abs(specification.ModelFile)
		if err != nil {
			return err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: identity, Size: size})
		batch.Locations = append(batch.Locations, artifact.LocationEvent{
			Location: artifact.Location{Artifact: identity, Kind: artifact.LocationFile, Value: absolute},
			Action:   artifact.LocationAdd,
		})
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	committed, external := 0, 0
	for evidence := range claimed {
		if _, ok, err := artifact.ReadContent(ctx, store, evidence); err != nil {
			return err
		} else if ok {
			committed++
		} else {
			external++
		}
	}
	fmt.Fprintf(output, "model verification committed: %s model=%s name=%s claims=%d\n",
		record.ID, record.Model, record.Name, len(record.Claims))
	for _, claim := range record.Claims {
		provenance := "commit=" + claim.Commit
		if claim.Dataset.Valid() {
			provenance += fmt.Sprintf(" dataset=%s span_steps=%d span_tokens=%d", claim.Dataset, claim.SpanSteps, claim.SpanTokens)
		}
		if claim.WallNS > 0 {
			provenance += fmt.Sprintf(" wall=%s context_tokens=%d peak_device_bytes=%d",
				time.Duration(claim.WallNS).Round(time.Millisecond), claim.ContextTokens, claim.PeakDeviceBytes)
		}
		fmt.Fprintf(output, "claim %s: %s (%d evidence) %s\n", claim.Capability, claim.Tier, len(claim.Evidence), provenance)
	}
	fmt.Fprintf(output, "audit: %d evidence artifact(s) committed in this store, %d identified externally; tiers claim only what their evidence grounds\n",
		committed, external)
	return nil
}
