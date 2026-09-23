package webuilane

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	// StylesheetPath is the workbench stylesheet, the only source of its design tokens.
	StylesheetPath = "internal/server/webui/style.css"
	// DesignDocumentPath is the document derived from the stylesheet.
	DesignDocumentPath = "docs/workbench_design.json"
)

// The text tokens and the surfaces they are read on; the accent's own text
// token is read on the accent.
var (
	textTokens    = []string{"ink", "dim", "faint", "acc", "amber", "ok", "err"}
	surfaceTokens = []string{"bg0", "bg1", "bg2", "bg3"}
)

// DesignDocument is the workbench's design as style.css declares it: each
// colour scheme's colour tokens with the contrast of every text token on
// every surface, and the spacing, type and radius scales.
type DesignDocument struct {
	Source  string            `json:"source"`
	Schemes []DesignScheme    `json:"schemes"`
	Spacing map[string]string `json:"spacing"`
	Type    map[string]string `json:"type"`
	Radius  map[string]string `json:"radius"`
}

// DesignScheme is one colour scheme's tokens and measured pairs.
type DesignScheme struct {
	Name     string            `json:"name"`
	Colours  map[string]string `json:"colours"`
	Contrast []TokenContrast   `json:"contrast"`
}

// TokenContrast is one text token read on one surface.
type TokenContrast struct {
	Foreground string  `json:"foreground"`
	Background string  `json:"background"`
	Ratio      float64 `json:"ratio"`
}

var (
	rootBlock      = regexp.MustCompile(`(?s)(?:^|\n):root \{(.*?)\n\}`)
	lightRootBlock = regexp.MustCompile(`(?s)@media \(prefers-color-scheme: light\) \{\s*:root \{(.*?)\n  \}`)
	customProperty = regexp.MustCompile(`--([a-z0-9-]+):\s*([^;]*\S)\s*$`)
	hexColour      = regexp.MustCompile(`^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$`)
	spacingToken   = regexp.MustCompile(`^s[0-9]+$`)
)

// ParseDesign derives the design document from the stylesheet: the dark
// scheme is the root declaration, the light scheme overrides it under
// prefers-color-scheme: light.
func ParseDesign(stylesheet string) (DesignDocument, error) {
	dark := rootBlock.FindStringSubmatch(stylesheet)
	light := lightRootBlock.FindStringSubmatch(stylesheet)
	if dark == nil || light == nil {
		return DesignDocument{}, errors.New("webui design: the stylesheet declares no root token block for each scheme")
	}
	base := properties(dark[1])
	document := DesignDocument{Source: StylesheetPath, Spacing: map[string]string{}, Type: map[string]string{}, Radius: map[string]string{}}
	for name, value := range base {
		switch {
		case spacingToken.MatchString(name):
			document.Spacing[name] = value
		case strings.HasPrefix(name, "t-"):
			document.Type[name] = value
		case strings.HasPrefix(name, "radius-"):
			document.Radius[name] = value
		}
	}
	overrides := properties(light[1])
	for _, scheme := range []struct {
		name   string
		tokens map[string]string
	}{{"dark", base}, {"light", merged(base, overrides)}} {
		measured, err := measureScheme(scheme.name, scheme.tokens)
		if err != nil {
			return DesignDocument{}, err
		}
		document.Schemes = append(document.Schemes, measured)
	}
	return document, nil
}

// EncodeDesign renders the document as it is published: indented JSON
// with a trailing newline, keys in a stable order.
func EncodeDesign(document DesignDocument) ([]byte, error) {
	encoded, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(encoded, '\n'), nil
}

func properties(block string) map[string]string {
	values := map[string]string{}
	for declaration := range strings.SplitSeq(block, ";") {
		if match := customProperty.FindStringSubmatch(declaration); match != nil {
			values[match[1]] = match[2]
		}
	}
	return values
}

func merged(base, overrides map[string]string) map[string]string {
	values := maps.Clone(base)
	maps.Copy(values, overrides)
	return values
}

func measureScheme(name string, tokens map[string]string) (DesignScheme, error) {
	scheme := DesignScheme{Name: name, Colours: map[string]string{}}
	for token, value := range tokens {
		if hexColour.MatchString(value) {
			scheme.Colours[token] = strings.ToLower(value)
		}
	}
	pairs := make([][2]string, 0, len(textTokens)*len(surfaceTokens)+1)
	for _, foreground := range textTokens {
		for _, background := range surfaceTokens {
			pairs = append(pairs, [2]string{foreground, background})
		}
	}
	pairs = append(pairs, [2]string{"on-accent", "acc"})
	for _, pair := range pairs {
		ratio, err := contrastRatio(scheme.Colours[pair[0]], scheme.Colours[pair[1]])
		if err != nil {
			return DesignScheme{}, fmt.Errorf("webui design: %s %s on %s: %w", name, pair[0], pair[1], err)
		}
		scheme.Contrast = append(scheme.Contrast, TokenContrast{Foreground: pair[0], Background: pair[1], Ratio: ratio})
	}
	slices.SortFunc(scheme.Contrast, func(a, b TokenContrast) int {
		return strings.Compare(a.Foreground+"/"+a.Background, b.Foreground+"/"+b.Background)
	})
	return scheme, nil
}

// contrastRatio is the WCAG ratio of two hex colours, rounded to hundredths
// so the derived document is stable.
func contrastRatio(foreground, background string) (float64, error) {
	a, err := relativeLuminance(foreground)
	if err != nil {
		return 0, err
	}
	b, err := relativeLuminance(background)
	if err != nil {
		return 0, err
	}
	return math.Round((max(a, b)+0.05)/(min(a, b)+0.05)*100) / 100, nil
}

func relativeLuminance(colour string) (float64, error) {
	if !hexColour.MatchString(colour) {
		return 0, fmt.Errorf("%q is not a hex colour", colour)
	}
	digits := colour[1:]
	if len(digits) == 3 {
		digits = string([]byte{digits[0], digits[0], digits[1], digits[1], digits[2], digits[2]})
	}
	var luminance float64
	for index, weight := range []float64{0.2126, 0.7152, 0.0722} {
		value, err := strconv.ParseUint(digits[index*2:index*2+2], 16, 8)
		if err != nil {
			return 0, err
		}
		channel := float64(value) / 255
		if channel <= 0.03928 {
			channel /= 12.92
		} else {
			channel = math.Pow((channel+0.055)/1.055, 2.4)
		}
		luminance += weight * channel
	}
	return luminance, nil
}
