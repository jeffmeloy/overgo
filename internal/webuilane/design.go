package webuilane

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
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

// The text tokens and the opaque surfaces they may be read on, every pair of
// which is measured; the pairs the stylesheet paints elsewhere (the accent's
// text on the accent, ink on the translucent highlight) are measured beside.
var (
	textTokens    = []string{"ink", "dim", "faint", "acc", "amber", "ok", "err"}
	surfaceTokens = []string{"bg0", "bg1", "bg2", "bg3"}
)

// DesignDocument is the workbench's design as style.css declares it: each
// colour scheme's colour tokens with the contrast of every text token on
// every surface and of every pair the stylesheet paints, and the spacing,
// type and radius scales.
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

// The rules that declare the two schemes' tokens.
const (
	rootPrelude        = ":root"
	lightSchemePrelude = "@media (prefers-color-scheme: light)"
)

// ParseDesign derives the design document from the stylesheet: the dark
// scheme is the top-level root rule, the light scheme overrides it in the
// root rule under prefers-color-scheme: light.
func ParseDesign(stylesheet string) (DesignDocument, error) {
	sheet := cssRule{rules: parseCSS(stylesheet)}
	dark, darkFound := sheet.rule(rootPrelude)
	media, mediaFound := sheet.rule(lightSchemePrelude)
	light, lightFound := media.rule(rootPrelude)
	if !darkFound || !mediaFound || !lightFound {
		return DesignDocument{}, errors.New("webui design: the stylesheet declares no root token block for each scheme")
	}
	base := properties(dark)
	document := DesignDocument{Source: StylesheetPath, Spacing: map[string]string{}, Type: map[string]string{}, Radius: map[string]string{}}
	for name, value := range base {
		switch {
		case spacingToken(name):
			document.Spacing[name] = value
		case strings.HasPrefix(name, "t-"):
			document.Type[name] = value
		case strings.HasPrefix(name, "radius-"):
			document.Radius[name] = value
		}
	}
	overrides := properties(light)
	painted := paintedPairs(sheet)
	for _, scheme := range []struct {
		name   string
		tokens map[string]string
	}{{"dark", base}, {"light", merged(base, overrides)}} {
		measured, err := measureScheme(scheme.name, scheme.tokens, painted)
		if err != nil {
			return DesignDocument{}, err
		}
		document.Schemes = append(document.Schemes, measured)
	}
	return document, nil
}

// paintedPairs answers each token text colour the stylesheet paints on a
// token background within one rule, at any nesting, in the order the rules
// declare them.
func paintedPairs(sheet cssRule) [][2]string {
	token := func(value string) string {
		name, ok := strings.CutPrefix(value, "var(--")
		if name, closed := strings.CutSuffix(name, ")"); ok && closed {
			return name
		}
		return ""
	}
	var pairs [][2]string
	sheet.walk(func(rule cssRule) {
		var foreground, background string
		for _, declaration := range rule.declarations {
			switch declaration.property {
			case "color":
				foreground = token(declaration.value)
			case "background", "background-color":
				background = token(declaration.value)
			}
		}
		if pair := [2]string{foreground, background}; foreground != "" && background != "" && !slices.Contains(pairs, pair) {
			pairs = append(pairs, pair)
		}
	})
	return pairs
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

// properties answers a rule's custom properties by name, without their
// leading dashes.
func properties(rule cssRule) map[string]string {
	values := map[string]string{}
	for _, declaration := range rule.declarations {
		if name, custom := strings.CutPrefix(declaration.property, "--"); custom && name != "" && declaration.value != "" {
			values[name] = declaration.value
		}
	}
	return values
}

// spacingToken: an s followed by digits names a spacing step.
func spacingToken(name string) bool {
	step, found := strings.CutPrefix(name, "s")
	return found && step != "" && strings.Trim(step, "0123456789") == ""
}

func merged(base, overrides map[string]string) map[string]string {
	values := maps.Clone(base)
	maps.Copy(values, overrides)
	return values
}

func measureScheme(name string, tokens map[string]string, painted [][2]string) (DesignScheme, error) {
	scheme := DesignScheme{Name: name, Colours: map[string]string{}}
	colours := map[string]colour{}
	for token, value := range tokens {
		if parsed, ok := parseColour(value); ok {
			colours[token], scheme.Colours[token] = parsed, strings.ToLower(strings.TrimSpace(value))
		}
	}
	pairs := make([][2]string, 0, len(textTokens)*len(surfaceTokens)+len(painted))
	for _, foreground := range textTokens {
		for _, background := range surfaceTokens {
			pairs = append(pairs, [2]string{foreground, background})
		}
	}
	for _, pair := range painted {
		if !slices.Contains(pairs, pair) {
			pairs = append(pairs, pair)
		}
	}
	for _, pair := range pairs {
		foreground, okForeground := colours[pair[0]]
		background, okBackground := colours[pair[1]]
		if !okForeground || !okBackground {
			return DesignScheme{}, fmt.Errorf("webui design: %s %s on %s: a token is not a colour", name, pair[0], pair[1])
		}
		ratio := contrast(foreground, background)
		// A translucent background is read over whichever surface lies under
		// it, so its pair counts at the worst of them.
		if background.alpha < 1 {
			ratio = math.Inf(1)
			for _, surface := range surfaceTokens {
				ratio = min(ratio, contrast(foreground, background.over(colours[surface])))
			}
		}
		scheme.Contrast = append(scheme.Contrast, TokenContrast{Foreground: pair[0], Background: pair[1], Ratio: ratio})
	}
	slices.SortFunc(scheme.Contrast, func(a, b TokenContrast) int {
		return strings.Compare(a.Foreground+"/"+a.Background, b.Foreground+"/"+b.Background)
	})
	return scheme, nil
}

// colour is one token colour: its channels on the 0-255 scale and its
// alpha, which is whole for an opaque colour.
type colour struct {
	channels [3]float64
	alpha    float64
}

// parseColour reads a hex colour or an rgb()/rgba() colour; any other
// value (a shadow, a length) names no colour.
func parseColour(value string) (colour, bool) {
	value = strings.ToLower(strings.TrimSpace(value))
	parsed := colour{alpha: 1}
	if digits, hexadecimal := strings.CutPrefix(value, "#"); hexadecimal {
		// The short form writes each channel's digit once.
		if len(digits) == len(parsed.channels) {
			digits = string([]byte{digits[0], digits[0], digits[1], digits[1], digits[2], digits[2]})
		}
		decoded, err := hex.DecodeString(digits)
		if err != nil || len(decoded) != len(parsed.channels) {
			return colour{}, false
		}
		for index, channel := range decoded {
			parsed.channels[index] = float64(channel)
		}
		return parsed, true
	}
	inner, functional := strings.CutPrefix(value, "rgba(")
	if !functional {
		inner, functional = strings.CutPrefix(value, "rgb(")
	}
	inner, closed := strings.CutSuffix(inner, ")")
	parts := strings.Split(inner, ",")
	if !functional || !closed || len(parts) < len(parsed.channels) || len(parts) > len(parsed.channels)+1 {
		return colour{}, false
	}
	for index, part := range parts {
		number, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return colour{}, false
		}
		if index < len(parsed.channels) {
			parsed.channels[index] = number
		} else {
			parsed.alpha = number
		}
	}
	return parsed, true
}

// over is the colour seen where a translucent colour lies on an opaque one.
func (c colour) over(ground colour) colour {
	seen := colour{alpha: ground.alpha}
	for index := range seen.channels {
		seen.channels[index] = c.channels[index]*c.alpha + ground.channels[index]*(1-c.alpha)
	}
	return seen
}

// contrast is the WCAG ratio of two colours, rounded to hundredths so the
// derived document is stable.
func contrast(foreground, background colour) float64 {
	a, b := relativeLuminance(foreground), relativeLuminance(background)
	return math.Round((max(a, b)+0.05)/(min(a, b)+0.05)*100) / 100
}

func relativeLuminance(c colour) float64 {
	var luminance float64
	for index, weight := range []float64{0.2126, 0.7152, 0.0722} {
		channel := c.channels[index] / 255
		if channel <= 0.03928 {
			channel /= 12.92
		} else {
			channel = math.Pow((channel+0.055)/1.055, 2.4)
		}
		luminance += weight * channel
	}
	return luminance
}
