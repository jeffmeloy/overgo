package modeldevice

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"

	"overgo/internal/atomicfile"
	"overgo/internal/cuda/device"
)

// ExpertProfile ranks a model's (layer, expert) pairs, the most routed first, as Strata's expert profile does
// (tools/make_profile.py, data/expert-profile.bin): a load places each layer's top-ranked experts on the card while its
// limit allows, where an index-order fill left the experts a request routes to chance (Flash-Next's agent turns kept
// 79 to 94% of their routed experts on the card, with Strata's ranking 90 to 96%). Its file is Strata's STRP version
// 1: "STRP", then version, layers, experts, slots and the ranked count as little-endian uint32, then each pair as two
// little-endian uint16, layer then expert, then each layer's experts' ranks as little-endian int32, -1 for one unranked.
type ExpertProfile struct {
	Layers, Experts uint32
	Pairs           []ExpertPair
}

// ExpertPair names one expert of one routed layer.
type ExpertPair struct {
	Layer, Expert uint16
}

// expertProfileHeader is a profile file's fields after its magic; Strata writes the ranked count as its slots too.
type expertProfileHeader struct {
	Version, Layers, Experts, Slots, Ranked uint32
}

const (
	expertProfileMagic   = "STRP"
	expertProfileVersion = 1
	// expertProfileFileMode lets every user read a written profile.
	expertProfileFileMode = 0o644
)

// ReadExpertProfile reads a profile file, refusing one whose pairs fall outside its own geometry.
func ReadExpertProfile(path string) (ExpertProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ExpertProfile{}, err
	}
	fields, found := bytes.CutPrefix(data, []byte(expertProfileMagic))
	var header expertProfileHeader
	read, err := binary.Decode(fields, binary.LittleEndian, &header)
	if !found || err != nil || header.Version != expertProfileVersion ||
		uint64(header.Ranked)*uint64(binary.Size(ExpertPair{})) > uint64(len(fields)-read) {
		return ExpertProfile{}, fmt.Errorf("expert profile %s is no STRP version %d file holding its pairs", path, expertProfileVersion)
	}
	profile := ExpertProfile{Layers: header.Layers, Experts: header.Experts, Pairs: make([]ExpertPair, header.Ranked)}
	if _, err := binary.Decode(fields[read:], binary.LittleEndian, profile.Pairs); err != nil {
		return ExpertProfile{}, err
	}
	for index, pair := range profile.Pairs {
		if uint32(pair.Layer) >= header.Layers || uint32(pair.Expert) >= header.Experts {
			return ExpertProfile{}, fmt.Errorf("expert profile %s pair %d is layer %d expert %d of %dx%d", path, index, pair.Layer, pair.Expert, header.Layers, header.Experts)
		}
	}
	return profile, nil
}

// Write writes the profile's file.
func (p ExpertProfile) Write(path string) error {
	ranks := make([]int32, uint64(p.Layers)*uint64(p.Experts))
	for index := range ranks {
		ranks[index] = -1
	}
	for rank, pair := range p.Pairs {
		ranks[uint64(pair.Layer)*uint64(p.Experts)+uint64(pair.Expert)] = int32(rank)
	}
	ranked := uint32(len(p.Pairs))
	data := bytes.NewBufferString(expertProfileMagic)
	for _, part := range []any{expertProfileHeader{expertProfileVersion, p.Layers, p.Experts, ranked, ranked}, p.Pairs, ranks} {
		if err := binary.Write(data, binary.LittleEndian, part); err != nil {
			return err
		}
	}
	return atomicfile.Write(path, data.Bytes(), expertProfileFileMode)
}

// rankExpertsByRoutes ranks every (layer, expert) pair by the runs that routed it, the most first, then each never
// routed in turn across the layers, as Strata's make_profile.py appends the missing pairs; it refuses layers or experts
// beyond the indices a pair's uint16 fields name.
func rankExpertsByRoutes(layers, experts uint32, routes func(layer, expert uint32) uint64) (ExpertProfile, error) {
	if max(layers, experts) > math.MaxUint16+1 {
		return ExpertProfile{}, fmt.Errorf("an expert profile's pairs name at most %d layers and experts, not %d layers of %d", math.MaxUint16+1, layers, experts)
	}
	type counted struct {
		pair  ExpertPair
		count uint64
	}
	var routed []counted
	var missing []ExpertPair
	for expert := range experts {
		for layer := range layers {
			pair := ExpertPair{uint16(layer), uint16(expert)}
			if count := routes(layer, expert); count != 0 {
				routed = append(routed, counted{pair, count})
			} else {
				missing = append(missing, pair)
			}
		}
	}
	slices.SortStableFunc(routed, func(a, b counted) int { return cmp.Compare(b.count, a.count) })
	profile := ExpertProfile{Layers: layers, Experts: experts, Pairs: make([]ExpertPair, 0, len(routed)+len(missing))}
	for _, item := range routed {
		profile.Pairs = append(profile.Pairs, item.pair)
	}
	profile.Pairs = append(profile.Pairs, missing...)
	return profile, nil
}

// expertBankLayer reads the routed layer a bank's GGUF name ("blk.N.") names.
func expertBankLayer(bank *expertBank) (uint32, error) {
	var layer uint32
	if _, err := fmt.Sscanf(bank.info.Name, "blk.%d.", &layer); err != nil {
		return 0, fmt.Errorf("expert bank %q names no layer: %w", bank.info.Name, err)
	}
	return layer, nil
}

// rankedCardExperts marks the experts a profile places on the card: its pairs in order, each layer's banks together
// (its blob), while the limit allows; it answers each blob's marks.
func (c *expertResidency) rankedCardExperts(profile ExpertProfile, limit uint64) (map[*expertBlob][]bool, error) {
	byLayer := map[uint32][]*expertBlob{}
	marks := make(map[*expertBlob][]bool, len(c.blobs))
	for _, blob := range c.blobs {
		for _, bank := range blob.banks {
			if uint32(len(bank.pointers)) != profile.Experts {
				return nil, fmt.Errorf("expert profile ranks %d experts a layer, bank %q holds %d", profile.Experts, bank.info.Name, len(bank.pointers))
			}
		}
		layer, err := expertBankLayer(blob.head())
		if err != nil {
			return nil, err
		}
		if layer >= profile.Layers {
			return nil, errors.New("expert profile ranks fewer layers than the banks name")
		}
		byLayer[layer] = append(byLayer[layer], blob)
		marks[blob] = make([]bool, len(blob.head().pointers))
	}
	for _, blob := range c.blobs {
		blob.rank = make([]uint32, len(blob.head().pointers))
		for index := range blob.rank {
			blob.rank[index] = uint32(len(profile.Pairs))
		}
	}
	for place, pair := range profile.Pairs {
		for _, blob := range byLayer[uint32(pair.Layer)] {
			blob.rank[pair.Expert] = min(blob.rank[pair.Expert], uint32(place))
			if !marks[blob][pair.Expert] && blob.stride <= limit-c.bytes {
				marks[blob][pair.Expert] = true
				c.bytes += blob.stride
			}
		}
	}
	return marks, nil
}

// ExpertProfile ranks the residency's (layer, expert) pairs by the decode runs that routed them, on its owning worker:
// the profile a load given it fills the card from.
func (w *DeviceWeights) ExpertProfile(state *device.State) (ExpertProfile, error) {
	if w == nil || w.experts == nil || w.experts.state != state {
		return ExpertProfile{}, errors.New("expert profile requires its owning worker")
	}
	runs := map[uint32][]uint64{}
	var layers, experts uint32
	for _, head := range w.experts.ordered {
		// A routed node's first bank counts its runs, as only its route flags are set.
		if head.group == nil {
			continue
		}
		layer, err := expertBankLayer(head)
		if err != nil {
			return ExpertProfile{}, err
		}
		if experts != 0 && experts != uint32(len(head.pointers)) {
			return ExpertProfile{}, fmt.Errorf("expert bank %q holds %d experts where another holds %d", head.info.Name, len(head.pointers), experts)
		}
		experts, layers, runs[layer] = uint32(len(head.pointers)), max(layers, layer+1), head.decodeRuns
	}
	return rankExpertsByRoutes(layers, experts, func(layer, expert uint32) uint64 {
		if counts := runs[layer]; counts != nil {
			return counts[expert]
		}
		return 0
	})
}
