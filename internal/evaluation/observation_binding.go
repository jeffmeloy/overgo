package evaluation

import (
	"cmp"
	"encoding/json"
	"fmt"

	"overgo/internal/strictjson"
)

// Observation fields are what an instrument other than the judge read of a suite's media: an OCR reader's readings of
// its pictures (PictureReadings) and an image encoder's motion of its clips (ClipMotion).
const (
	readingsField = "readings"
	motionField   = "motion"
)

// observationFields are the suite fields that hold observations of its media, bound to the media by bindObservations.
var observationFields = []string{readingsField, motionField}

// judgedMedia is a case's media as its suite names it and as the compile bound it: the reference its bound prompt
// names, the digest of the content it consumes, and what the case's clip is measured by (MotionKey).
type judgedMedia struct {
	kind                     MediaKind
	name, reference, content string
	motionKey                string
}

// motionKey is a case's MotionKey read from its fields: its group, or its name where it has none.
func motionKey(testCase map[string]json.RawMessage) string {
	var name, group string
	_ = json.Unmarshal(testCase["name"], &name)
	_ = json.Unmarshal(testCase["group"], &group)
	return cmp.Or(group, name)
}

// bindObservations puts a suite's observations back into its bound document, each bound to the media its cases judge:
// readings re-keyed from the picture's name to the reference the bound cases name it by, motion as it is keyed. An
// observation set any member of which read other content than the case's media consumes, a picture or clip changed
// since it was read or a measure that names no content, is held out whole, so no instrument's reading is credited to
// media it did not read and none is averaged over part of a suite. The binding reads the content alone, apart from
// the answering runtime the reference also names: the instrument that observed is not the judge.
func bindObservations(document, observed map[string]json.RawMessage, judged []judgedMedia) error {
	if raw, held := observed[readingsField]; held {
		var readings PictureReadings
		if err := strictjson.DecodeBytes(raw, &readings); err != nil {
			return err
		}
		names, bound, stale := map[string]bool{}, map[string]PictureReading{}, false
		for _, media := range judged {
			reading, read := readings.Read[media.name]
			if media.kind != MediaImage || !read {
				continue
			}
			names[media.name], bound[media.reference] = true, reading
			stale = stale || reading.Content != media.content
		}
		for name := range readings.Read {
			if !names[name] {
				return fmt.Errorf("evaluation: the suite's readings read %s, which no case asks about", name)
			}
		}
		if !stale {
			readings.Read = bound
			encoded, err := json.Marshal(readings)
			if err != nil {
				return err
			}
			document[readingsField] = encoded
		}
	}
	if raw, held := observed[motionField]; held {
		var motion ClipMotion
		if err := strictjson.DecodeBytes(raw, &motion); err != nil {
			return err
		}
		stale := false
		for _, media := range judged {
			if clip, measured := motion.Clips[media.motionKey]; media.kind == MediaVideo && measured {
				stale = stale || clip.Content != media.content
			}
		}
		if !stale {
			document[motionField] = raw
		}
	}
	return nil
}
