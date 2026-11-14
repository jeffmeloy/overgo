package speechsynth

import (
	"path/filepath"
	"strings"

	"overgo/internal/jsonfile"
)

// defaultVoices are the reference's own voice for a language (pocket_tts default_parameters.py,
// DEFAULT_VOICE_FOR_LANGUAGE), in its order: `pocket-tts generate` reads a text that names no voice in it.
var defaultVoices = [][2]string{{"italian", "giovanni"}, {"spanish", "lola"}, {"german", "juergen"}, {"portuguese", "rafael"}, {"french", "estelle"}}

// fallbackVoice is the reference's voice for any other language (DEFAULT_VOICE_FALLBACK).
const fallbackVoice = "alba"

// DefaultVoice is the voice the reference reads a request in that names no voice, no prepared voice and no reference
// recording, for the language the model's weights are of: the language its config's weights path names
// (languages/<language>/...), matched as the reference matches it, by a language the table names within it, and the
// fallback where none is named or matched (declared-defaults/do). A person who asked for speech and chose no voice was
// refused, the page offering the model's voices with none chosen.
func DefaultVoice(directory string) (string, error) {
	var config struct {
		Weights string `json:"weights_path_without_voice_cloning"`
	}
	if err := jsonfile.Decode(filepath.Join(directory, configFileName), &config); err != nil {
		return "", err
	}
	_, language, _ := strings.Cut(config.Weights, "languages/")
	language, _, _ = strings.Cut(language, "/")
	for _, entry := range defaultVoices {
		if language != "" && strings.Contains(language, entry[0]) {
			return entry[1], nil
		}
	}
	return fallbackVoice, nil
}
