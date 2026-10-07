package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/composition"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/rae"
	"overgo/internal/tensor"
	"overgo/internal/vitencoder"
)

const (
	// bridgeFrameFPS is the rate a clip is sampled at for its first frame, the rate the
	// reconstruction suites cut at, so a frame here is the frame a suite would show.
	bridgeFrameFPS = 16
	// bridgeCacheMode and bridgeCacheFileMode keep a run's cached tokens to its owner.
	bridgeCacheMode     = 0o700
	bridgeCacheFileMode = 0o600
	// bridgeClipExtension is the file extension of a clip in a clip directory.
	bridgeClipExtension = ".mp4"
)

// bridgeFrame is one picture, channel-first in [0, 1], with the tokens the two encoders
// give for it: the source encoder's, which a bridge maps, and the decoder's own
// encoder's, which it is fitted to.
type bridgeFrame struct {
	picture []float32
	source  []float32
	target  []float32
}

// bridgeFit is a bridge fitted in closed form: the best affine map from the source
// tokens to the target tokens of the frames it was given, the share of the target it
// leaves unexplained on those frames, and the adapter rung that share admits.
type bridgeFit struct {
	Bridge   rae.EncoderBridge
	Residual float64
	Rung     composition.AdapterRung
}

// bridgeVerdict is a bridge judged on frames it was not fitted to: the mean peak
// signal-to-noise ratio of what the decoder draws from its own encoder's tokens and
// from the bridged ones, the mean of the bridged less the own frame by frame with its
// standard error, and the share of the target tokens the bridge leaves unexplained.
type bridgeVerdict struct {
	Frames          int     `json:"frames"`
	OwnEncoderDB    float64 `json:"own_encoder_psnr_db"`
	BridgedDB       float64 `json:"bridged_psnr_db"`
	DifferenceDB    float64 `json:"bridged_less_own_db"`
	DifferenceError float64 `json:"bridged_less_own_standard_error_db"`
	FeatureResidual float64 `json:"feature_residual"`
}

// fitEncoderBridge fits the affine bridge over every token of the frames: the seam
// alignment's closed-form linear map, with a constant beside each source token so the
// map carries an offset. It refuses frames too few to determine the map.
func fitEncoderBridge(frames []bridgeFrame, sourceWidth, targetWidth int) (bridgeFit, error) {
	var sourceRows, targetRows [][]float64
	for _, frame := range frames {
		if len(frame.source)%sourceWidth != 0 || len(frame.target)%targetWidth != 0 || len(frame.source)/sourceWidth != len(frame.target)/targetWidth {
			return bridgeFit{}, errors.New("encoder bridge: a frame's two token grids differ")
		}
		for token := range len(frame.source) / sourceWidth {
			row := make([]float64, sourceWidth+tensor.SingletonExtent)
			for at, value := range frame.source[token*sourceWidth:][:sourceWidth] {
				row[at] = float64(value)
			}
			row[sourceWidth] = tensor.SingletonExtent
			want := make([]float64, targetWidth)
			for at, value := range frame.target[token*targetWidth:][:targetWidth] {
				want[at] = float64(value)
			}
			sourceRows, targetRows = append(sourceRows, row), append(targetRows, want)
		}
	}
	if len(sourceRows) <= sourceWidth+tensor.SingletonExtent {
		return bridgeFit{}, fmt.Errorf("encoder bridge: %d tokens cannot determine a map %d wide", len(sourceRows), sourceWidth+tensor.SingletonExtent)
	}
	// An encoder's tokens leave its final norm, which fixes a sum over their channels, so
	// the tokens and the constant beside them are exactly dependent in one direction and
	// the plain fit is undetermined there. The tokens are single-precision: a direction
	// whose energy is under their rounding floor carries rounding and nothing else. Each
	// source direction is therefore given that floor of energy mapped to nothing, which
	// settles such a direction at no weight and leaves every measured direction alone.
	var scale float64
	for column := range sourceWidth + tensor.SingletonExtent {
		var energy float64
		for _, row := range sourceRows {
			energy += row[column] * row[column]
		}
		scale = max(scale, energy)
	}
	floor := math.Sqrt(scale * float64(math.Nextafter32(tensor.SingletonExtent, tensor.PairedExtent)-tensor.SingletonExtent))
	for column := range sourceWidth + tensor.SingletonExtent {
		row := make([]float64, sourceWidth+tensor.SingletonExtent)
		row[column] = floor
		sourceRows, targetRows = append(sourceRows, row), append(targetRows, make([]float64, targetWidth))
	}
	adapter, residual, err := composition.AlignSeamAdapter(sourceRows, targetRows)
	if err != nil {
		return bridgeFit{}, err
	}
	// The adapter holds a row of target weights for each source channel and, last, the
	// offsets; a bridge holds a row of source weights for each target channel.
	bridge := rae.EncoderBridge{Weight: make([]float32, targetWidth*sourceWidth), Bias: make([]float32, targetWidth)}
	for channel := range targetWidth {
		for at := range sourceWidth {
			bridge.Weight[channel*sourceWidth+at] = float32(adapter[at][channel])
		}
		bridge.Bias[channel] = float32(adapter[sourceWidth][channel])
	}
	return bridgeFit{Bridge: bridge, Residual: residual, Rung: composition.AdapterRungFor(residual)}, nil
}

// judgeEncoderBridge has the pair's decoder draw each frame twice, from its own
// encoder's tokens and from the source encoder's tokens through the bridge, and
// compares both with the frame.
func judgeEncoderBridge(ctx context.Context, pair *rae.Autoencoder, bridge rae.EncoderBridge, frames []bridgeFrame) (bridgeVerdict, error) {
	if len(frames) == 0 {
		return bridgeVerdict{}, errors.New("encoder bridge: no frames to judge on")
	}
	verdict := bridgeVerdict{Frames: len(frames)}
	differences := make([]float64, len(frames))
	var featureError, featureEnergy float64
	for index, frame := range frames {
		mapped, err := bridge.Apply(frame.source)
		if err != nil {
			return bridgeVerdict{}, err
		}
		for at, value := range frame.target {
			difference := float64(mapped[at]) - float64(value)
			featureError += difference * difference
			featureEnergy += float64(value) * float64(value)
		}
		var ratios [tensor.PairedExtent]float64
		for side, tokens := range [tensor.PairedExtent][]float32{frame.target, mapped} {
			drawn, err := pair.Decode(ctx, tokens)
			if err != nil {
				return bridgeVerdict{}, err
			}
			if ratios[side], err = evaluation.PeakSignalToNoise(signedUnit(frame.picture), signedUnit(drawn)); err != nil {
				return bridgeVerdict{}, err
			}
		}
		verdict.OwnEncoderDB += ratios[0] / float64(len(frames))
		verdict.BridgedDB += ratios[1] / float64(len(frames))
		differences[index] = ratios[1] - ratios[0]
		verdict.DifferenceDB += differences[index] / float64(len(frames))
	}
	var spread float64
	for _, difference := range differences {
		spread += (difference - verdict.DifferenceDB) * (difference - verdict.DifferenceDB)
	}
	if len(frames) > tensor.SingletonExtent {
		verdict.DifferenceError = math.Sqrt(spread / float64(len(frames)-tensor.SingletonExtent) / float64(len(frames)))
	}
	verdict.FeatureResidual = math.Sqrt(featureError / featureEnergy)
	return verdict, nil
}

// signedUnit maps values in [0, 1] to the [-1, 1] the judge's ratio reads.
func signedUnit(unit []float32) []float32 {
	signed := make([]float32, len(unit))
	for index, value := range unit {
		signed[index] = tensor.PairedExtent*value - tensor.SingletonExtent
	}
	return signed
}

// bridgeTokens computes one picture's two token grids.
func bridgeTokens(ctx context.Context, pair *rae.Autoencoder, source *vitencoder.Encoder, picture []float32) (bridgeFrame, error) {
	pixels, err := pair.EncoderInput(ctx, picture)
	if err != nil {
		return bridgeFrame{}, err
	}
	target, err := pair.Tokens(pixels)
	if err != nil {
		return bridgeFrame{}, err
	}
	sourceTokens, err := rae.BridgeSourceTokens(source, picture, pair.Side())
	if err != nil {
		return bridgeFrame{}, err
	}
	return bridgeFrame{picture: picture, source: sourceTokens, target: target}, nil
}

// clipBridgeFrame cuts a clip's first frame to the pair's size, keeping its proportions,
// and computes its tokens, or reads them from the cache when a run before this one
// computed them for a clip of the same bytes.
func clipBridgeFrame(ctx context.Context, pair *rae.Autoencoder, source *vitencoder.Encoder, clip, cache string) (bridgeFrame, error) {
	side := pair.Side()
	frames, err := media.DecodeVideoFileCovering(ctx, clip, os.Getenv("OVERGO_FFMPEG"), bridgeFrameFPS, tensor.SingletonExtent, side, side)
	if err != nil || len(frames) != tensor.SingletonExtent {
		return bridgeFrame{}, fmt.Errorf("encoder bridge: %s gave %d frames: %w", clip, len(frames), err)
	}
	video, err := latentvideo.SourceVideoFromFrames(frames)
	if err != nil {
		return bridgeFrame{}, err
	}
	picture := make([]float32, len(video.Pixels))
	for index, value := range video.Pixels {
		picture[index] = (value + tensor.SingletonExtent) / tensor.PairedExtent
	}
	data, err := os.ReadFile(clip)
	if err != nil {
		return bridgeFrame{}, err
	}
	sum := sha256.Sum256(data)
	_, sourceGrid, sourceWidth := source.Shape()
	targetWidth, targetGrid := pair.Latent()
	sourceValues, targetValues := sourceGrid*sourceGrid*sourceWidth, targetGrid*targetGrid*targetWidth
	cached := filepath.Join(cache, hex.EncodeToString(sum[:])+".tokens")
	if raw, err := os.ReadFile(cached); err == nil && len(raw) == (sourceValues+targetValues)*binary.Size(float32(0)) {
		values := make([]float32, sourceValues+targetValues)
		for index := range values {
			values[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*binary.Size(float32(0)):]))
		}
		return bridgeFrame{picture: picture, source: values[:sourceValues], target: values[sourceValues:]}, nil
	}
	frame, err := bridgeTokens(ctx, pair, source, picture)
	if err != nil {
		return bridgeFrame{}, err
	}
	raw := make([]byte, 0, (sourceValues+targetValues)*binary.Size(float32(0)))
	for _, value := range slices.Concat(frame.source, frame.target) {
		raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(value))
	}
	return frame, os.WriteFile(cached, raw, bridgeCacheFileMode)
}

// trainEncoderBridge is the encoder-bridge route: it fits the affine map that lets one
// encoder's tokens feed a representation autoencoder's decoder, judges it on frames the
// fit never saw by what the decoder then draws, beside the decoder's own encoder, and
// writes the bridge as a pair of its own.
func trainEncoderBridge(ctx context.Context, flags *flag.FlagSet, args []string) error {
	sourceDir := flags.String("source", "", "the encoder to bridge: a checkpoint directory beside the collection")
	collectionDir := flags.String("collection", "", "the representation autoencoder's collection, whose decoder reads the bridged tokens")
	clips := flags.String("clips", "", "directory of clips whose first frames the bridge is fitted and judged on")
	exclude := flags.String("exclude", "", "a reconstruction suite whose clips are kept out of the fit and of the judging here, so the suite stays held out")
	fitFrames := flags.Int("fit-frames", 0, "frames the bridge is fitted on")
	judgeFrames := flags.Int("judge-frames", 0, "other frames the fitted bridge is judged on")
	seed := flags.Uint64("seed", 0, "the seed that picks the frames")
	cache := flags.String("cache", "", "directory that keeps each frame's tokens between runs")
	out := flags.String("out", "", "the bridged pair's directory, beside the source encoder and the collection")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *sourceDir == "" || *collectionDir == "" || *clips == "" || *cache == "" || *out == "" || *fitFrames <= 0 || *judgeFrames <= 0 {
		return fmt.Errorf("%s: source, collection, clips, cache, out, and positive fit-frames and judge-frames required", flags.Name())
	}
	sibling := func(directory string) string { return filepath.Dir(filepath.Clean(directory)) }
	if sibling(*sourceDir) != sibling(*collectionDir) || sibling(*out) != sibling(*collectionDir) {
		return fmt.Errorf("%s: the source encoder, the collection and the bridged pair sit beside each other", flags.Name())
	}
	held := map[string]bool{}
	if *exclude != "" {
		var suite evaluation.ReconstructionSuite
		if err := jsonfile.DecodeStrict(*exclude, &suite); err != nil {
			return err
		}
		for _, testCase := range suite.Cases {
			held[testCase.Clip] = true
		}
	}
	entries, err := os.ReadDir(*clips)
	if err != nil {
		return err
	}
	var names []string
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), bridgeClipExtension) && !held[entry.Name()] {
			names = append(names, entry.Name())
		}
	}
	slices.Sort(names)
	if len(names) < *fitFrames+*judgeFrames {
		return fmt.Errorf("%s: %s holds %d clips outside the excluded suite, fewer than the %d asked", flags.Name(), *clips, len(names), *fitFrames+*judgeFrames)
	}
	rng := rand.New(rand.NewPCG(*seed, *seed))
	rng.Shuffle(len(names), func(left, right int) { names[left], names[right] = names[right], names[left] })
	pair, err := rae.LoadAutoencoder(*collectionDir)
	if err != nil {
		return err
	}
	source, err := vitencoder.Load(*sourceDir)
	if err != nil {
		return err
	}
	cacheDirectory := filepath.Join(*cache, filepath.Base(filepath.Clean(*sourceDir))+"--"+filepath.Base(filepath.Clean(*collectionDir)))
	if err := os.MkdirAll(cacheDirectory, bridgeCacheMode); err != nil {
		return err
	}
	frames := make([]bridgeFrame, *fitFrames+*judgeFrames)
	for index := range frames {
		if frames[index], err = clipBridgeFrame(ctx, pair, source, filepath.Join(*clips, names[index]), cacheDirectory); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "  tokens %d/%d\n", index+tensor.SingletonExtent, len(frames))
	}
	_, _, sourceWidth := source.Shape()
	targetWidth, _ := pair.Latent()
	fit, err := fitEncoderBridge(frames[:*fitFrames], sourceWidth, targetWidth)
	if err != nil {
		return err
	}
	verdict, err := judgeEncoderBridge(ctx, pair, fit.Bridge, frames[*fitFrames:])
	if err != nil {
		return err
	}
	if err := rae.SaveEncoderBridge(*out, fit.Bridge, filepath.Base(filepath.Clean(*sourceDir)), filepath.Base(filepath.Clean(*collectionDir))); err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(struct {
		FitFrames   int                     `json:"fit_frames"`
		FitResidual float64                 `json:"fit_residual"`
		Rung        composition.AdapterRung `json:"rung"`
		Judged      bridgeVerdict           `json:"judged"`
		Out         string                  `json:"out"`
	}{FitFrames: *fitFrames, FitResidual: fit.Residual, Rung: fit.Rung, Judged: verdict, Out: *out})
}
