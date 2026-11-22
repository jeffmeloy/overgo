// Command speech-capture remakes a speech capture whose tensors are gone, from overgo's own speech path at a commit
// where the speech acceptances passed against it: the CPU ASR reference TestCPUASRReference reads (-asr), or one of the
// transducer captures the transducer acceptances read (-transducer with -capture). It writes the capture's tensors
// beneath the data root's references, registers their file in the store, publishes the capture naming overgo at
// -commit as its maker, and rewrites the record the acceptances read: the encoder capture record, or the transducer
// list with the remade capture in place of the one it remakes.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/audioparity"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
)

// speechPaths are the speech path's packages and declarations, whose source at -commit is the source that runs: a
// remake refuses a tree where any of them differs from that commit's, so the maker it records is the code that ran.
var speechPaths = []string{
	"internal/audiodsp", "internal/audioparity/recipes", "internal/dataset", "internal/hfbpe",
	"internal/media", "internal/safetensors", "internal/speechrecognition",
}

func main() { clioptions.MainWithArguments("speech-capture", run) }

// remakeRequest is one remake as the flags name it.
type remakeRequest struct {
	store, dataRoot, commit string
	asr, election           string
	transducer, capture     string
}

// usage is the command's one line of use.
const usage = "usage: speech-capture -commit <full commit> [-repo <store>] [-data-root <directory>] " +
	"(-asr <encoder capture record> -election <election record> | -transducer <transducer capture list> -capture <capture id>)"

func run(args []string, output io.Writer) error {
	request, err := parse(args)
	if err != nil {
		return err
	}
	ctx := context.Background()
	root, err := gitauthority.RepositoryRoot(ctx, ".")
	if err != nil {
		return err
	}
	if err := speechCodeAt(ctx, root, request.commit); err != nil {
		return err
	}
	resolve := dataroot.ResolveCurrent
	if request.dataRoot != "" {
		resolve = func() (dataroot.Roots, error) { return dataroot.Resolve(request.dataRoot) }
	}
	roots, err := resolve()
	if err != nil {
		return err
	}
	storeRoot := request.store
	if storeRoot == "" {
		storeRoot = roots.Store
	}
	store, err := overgodb.Open(storeRoot)
	if err != nil {
		return err
	}
	remake := audioparity.SpeechCaptureRemake{Store: store, Datasets: roots.Datasets, References: roots.References(), Commit: request.commit}
	id, err := remakeOne(ctx, remake, request)
	if err = errors.Join(err, store.Close()); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "speech capture=%s made by overgo at %s through %s; tensors under %s\n", id, request.commit, audioparity.SpeechCaptureCommand, roots.References())
	return err
}

// parse reads the flags into one remake, refusing a request that names neither kind of capture or both, a capture
// without the record it rewrites, and a commit that is not a full lowercase object name.
func parse(args []string) (remakeRequest, error) {
	flags := flag.NewFlagSet("speech-capture", flag.ContinueOnError)
	var request remakeRequest
	flags.StringVar(&request.store, "repo", "", "OvergoDB store the capture's inputs are recorded in and the remade capture is published to; empty resolves via the data-root contract")
	flags.StringVar(&request.dataRoot, "data-root", "", "data root whose datasets the corpus is read from and beside whose store the tensors are written, under references/speech-captures; empty resolves the current directory's")
	flags.StringVar(&request.commit, "commit", "", "full commit whose speech path makes the capture, one at which the speech acceptances passed against the capture remade")
	flags.StringVar(&request.asr, "asr", "", "the CPU ASR reference capture record to remake and rewrite (internal/audioparity/testdata/encoder_capture.json)")
	flags.StringVar(&request.election, "election", "", "the election record the CPU ASR reference binds (internal/audioparity/testdata/granite_speech_5_election.json)")
	flags.StringVar(&request.transducer, "transducer", "", "the transducer capture list to rewrite with the remade capture (internal/audioparity/testdata/transducer_captures.json)")
	flags.StringVar(&request.capture, "capture", "", "the transducer capture, by its evidence id, to remake")
	if err := flags.Parse(args); err != nil {
		return remakeRequest{}, err
	}
	asr, transducer := request.asr != "" || request.election != "", request.transducer != "" || request.capture != ""
	if flags.NArg() != 0 || asr == transducer || asr && (request.asr == "" || request.election == "") || transducer && (request.transducer == "" || request.capture == "") {
		return remakeRequest{}, errors.New(usage)
	}
	if !gitauthority.ValidObjectID(request.commit) || strings.ToLower(request.commit) != request.commit {
		return remakeRequest{}, fmt.Errorf("-commit %q is not a full lowercase commit: %s", request.commit, usage)
	}
	return request, nil
}

// remakeOne remakes the capture request names.
func remakeOne(ctx context.Context, remake audioparity.SpeechCaptureRemake, request remakeRequest) (artifact.ID, error) {
	if request.asr != "" {
		encoded, err := artifact.ReadContentFile(request.election)
		if err != nil {
			return artifact.ID{}, err
		}
		election, err := audioparity.NormalizeElection(encoded)
		if err != nil {
			return artifact.ID{}, err
		}
		return audioparity.RemakeEncoderCapture(ctx, remake, election, request.asr)
	}
	id, err := artifact.ParseID(request.capture)
	if err != nil {
		return artifact.ID{}, err
	}
	return audioparity.RemakeTransducerCapture(ctx, remake, request.transducer, id)
}

// speechCodeAt refuses a tree at root whose speech path differs from commit's, tracked or not, and a commit root's
// history does not hold.
func speechCodeAt(ctx context.Context, root, commit string) error {
	resolved, err := gitauthority.Query(ctx, root, "rev-parse", "--verify", "--quiet", commit+"^{commit}")
	if err != nil || strings.TrimSpace(string(resolved)) != commit {
		return errors.Join(fmt.Errorf("-commit %s is no commit of %s", commit, root), err)
	}
	changed, err := gitauthority.Query(ctx, root, append([]string{"diff", "--name-only", commit, "--"}, speechPaths...)...)
	if err != nil {
		return err
	}
	untracked, err := gitauthority.Query(ctx, root, append([]string{"ls-files", "--others", "--exclude-standard", "--"}, speechPaths...)...)
	if err != nil {
		return err
	}
	if differs := strings.Fields(string(changed) + string(untracked)); len(differs) != 0 {
		return fmt.Errorf("the speech path here differs from %s's, so overgo at %s did not make what this tree would: %s", commit, commit, strings.Join(differs, ", "))
	}
	return nil
}
