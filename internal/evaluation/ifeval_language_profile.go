package evaluation

import (
	"crypto/sha256"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"unicode/utf8"

	"github.com/dlclark/regexp2/v2"
)

//go:embed testdata/ifeval_langdetect_profile.json
var ifevalLanguageProfile []byte

const (
	ifevalLanguageRule = "lm-eval/ifeval/language/langdetect-1.0.9/v1"
	ifevalEnglishUpper = "lm-eval/ifeval/english-uppercase/langdetect-1.0.9/v1"
	ifevalEnglishLower = "lm-eval/ifeval/english-lowercase/langdetect-1.0.9/v1"
)

type ifevalLanguageInput struct {
	Languages []struct {
		Name  string
		Freq  map[string]int
		Words []int `json:"n_words"`
	}
	Ranges      [][3]int `json:"normalization_ranges"`
	Vietnamese  map[string]string
	State       []uint32 `json:"initial_random_state"`
	Declared    []string `json:"declared_languages"`
	Alpha       float64
	AlphaWidth  float64 `json:"alpha_width"`
	Iterations  int
	Threshold   float64 `json:"probability_threshold"`
	Convergence float64 `json:"convergence_threshold"`
	Frequency   float64 `json:"base_frequency"`
	Trials      int
	Maximum     int    `json:"max_text_length"`
	URL         string `json:"url_pattern"`
	Mail        string `json:"mail_pattern"`
	NGram       int    `json:"n_gram"`
}

type ifevalLanguageDetector struct {
	input     ifevalLanguageInput
	words     map[string][]float64
	url, mail *regexp2.Regexp
}

var compiledIFEvalLanguage = sync.OnceValues(func() (*ifevalLanguageDetector, error) { return loadIFEvalLanguage(ifevalLanguageProfile) })

func loadIFEvalLanguage(data []byte) (*ifevalLanguageDetector, error) {
	if fmt.Sprintf("%x", sha256.Sum256(data)) != "2b9af36fa26af89c5a87e5eae7b0b989236c06a8bb9f71f843d73e1676fc1c7b" {
		return nil, errors.New("evaluation: frozen language profile identity differs")
	}
	p := &ifevalLanguageDetector{words: map[string][]float64{}}
	if err := json.Unmarshal(data, &p.input); err != nil {
		return nil, err
	}
	for index, language := range p.input.Languages {
		for word, count := range language.Freq {
			values, found := p.words[word]
			if !found {
				values = make([]float64, len(p.input.Languages))
			}
			n := utf8.RuneCountInString(word)
			if n >= 1 && n <= p.input.NGram {
				values[index] = float64(count) / float64(language.Words[n-1])
			}
			p.words[word] = values
		}
	}
	var err error
	if p.url, err = regexp2.Compile(p.input.URL); err != nil {
		return nil, err
	}
	if p.mail, err = regexp2.Compile(p.input.Mail); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *ifevalLanguageDetector) random() ifevalPythonRandom {
	n := len(p.input.State) - 1
	return ifevalPythonRandom{state: slices.Clone(p.input.State[:n]), index: int(p.input.State[n])}
}

func (p *ifevalLanguageDetector) normalized(r rune) rune {
	index, found := slices.BinarySearchFunc(p.input.Ranges, int(r), func(value [3]int, goal int) int {
		if goal < value[0] {
			return 1
		}
		if goal > value[1] {
			return -1
		}
		return 0
	})
	if found {
		return rune(p.input.Ranges[index][2])
	}
	return r
}
