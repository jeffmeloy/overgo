package englishnorm

import (
	"fmt"
	"math/big"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/dlclark/regexp2/v2"
)

const (
	// tenUnit, hundredUnit and thousandUnit are the places a spoken number is
	// assembled by: a digit, a pair of digits and a group of three.
	tenUnit, hundredUnit, thousandUnit = 10, 100, 1000
	// joiningWord and pointWord are the two words the number reader's own
	// statements name: "and" after a multiplier is dropped, and "point" before
	// a number opens its decimals.
	joiningWord, pointWord = "and", "point"
	// halfAsDecimal and halfAsWords are what "and a half" becomes after a
	// number word and after any other word.
	halfAsDecimal, halfAsWords = "point five", "and a half"
	// centDigits is how many digits the cents of an amount are written with.
	centDigits = 2
	// currencyGroup, unitsGroup and centsGroup are what the pattern of an
	// amount with its cents captures, in its order: the currency's symbol,
	// the whole units and the cents.
	currencyGroup, unitsGroup, centsGroup = 1, 2, 3
)

// amount is the number being read: nothing yet, a whole number, or digits
// kept as they were written, which is how a number with a decimal point,
// leading zeros or digits read out one by one is carried.
type amount struct {
	held   bool
	whole  *big.Int
	digits string
}

func wholeAmount(value *big.Int) amount { return amount{held: true, whole: value} }

func writtenAmount(digits string) amount { return amount{held: true, digits: digits} }

// written reports whether the amount is carried as digits.
func (a amount) written() bool { return a.held && a.whole == nil }

func (a amount) text() string {
	if a.whole != nil {
		return a.whole.String()
	}
	return a.digits
}

// orEmpty is the amount's text to append to, where nothing read and a whole
// zero both start from nothing.
func (a amount) orEmpty() string {
	if !a.held || (a.whole != nil && a.whole.Sign() == 0) {
		return ""
	}
	return a.text()
}

// endsOn reports whether the whole amount's last places are zeros, so that a
// smaller number adds to it and is not read as further digits.
func (a amount) endsOn(unit int64) bool {
	return new(big.Int).Mod(a.whole, big.NewInt(unit)).Sign() == 0
}

// digit gives the value of a decimal digit of any script. Unicode encodes
// each script's ten digits together and in order from zero.
func digit(r rune) (int64, bool) {
	if !unicode.IsDigit(r) {
		return 0, false
	}
	for _, span := range unicode.Nd.R16 {
		if rune(span.Lo) <= r && r <= rune(span.Hi) {
			return int64(r-rune(span.Lo)) % tenUnit, true
		}
	}
	for _, span := range unicode.Nd.R32 {
		if rune(span.Lo) <= r && r <= rune(span.Hi) {
			return int64(r-rune(span.Lo)) % tenUnit, true
		}
	}
	return 0, false
}

// fraction reads digits with at most one decimal point as a number, the way
// that code's use of Python's Fraction reads the texts that reach it: "3."
// and ".5" are numbers, and "." and a text with two points are not. strict
// asks for digits on both sides of a point, as that code asks of a word
// before it calls it a number.
func fraction(text string, strict bool) (*big.Rat, bool) {
	numerator, scale := new(big.Int), big.NewInt(1)
	before, after, pointed := 0, 0, false
	for _, r := range text {
		value, isDigit := digit(r)
		switch {
		case isDigit:
			numerator.Mul(numerator, big.NewInt(tenUnit)).Add(numerator, big.NewInt(value))
			if pointed {
				scale.Mul(scale, big.NewInt(tenUnit))
				after++
			} else {
				before++
			}
		case r == '.' && !pointed:
			pointed = true
		default:
			return nil, false
		}
	}
	if before+after == 0 || (strict && (before == 0 || (pointed && after == 0))) {
		return nil, false
	}
	return new(big.Rat).SetFrac(numerator, scale), true
}

// scaled is the amount's text times a multiplier, when that is a whole
// number: "1.5" times a million is, and "1.5" times a hundred is too, where
// "0.001" times a hundred is not.
func scaled(text string, multiplier *big.Int) (*big.Int, bool) {
	value, ok := fraction(text, false)
	if !ok {
		return nil, false
	}
	product := value.Mul(value, new(big.Rat).SetInt(multiplier))
	if !product.IsInt() {
		return nil, false
	}
	return new(big.Int).Set(product.Num()), true
}

// grouped multiplies the last three places of a whole number and leaves the
// thousands above them: after "two thousand three", "hundred" makes two
// thousand three hundred.
func grouped(value, multiplier *big.Int) *big.Int {
	thousand := big.NewInt(thousandUnit)
	residual := new(big.Int)
	above, _ := new(big.Int).DivMod(value, thousand, residual)
	return above.Mul(above, thousand).Add(above, residual.Mul(residual, multiplier))
}

// trimLast drops a text's last character.
func trimLast(text string) string {
	_, size := utf8.DecodeLastRuneInString(text)
	return text[:len(text)-size]
}

// numbers writes the spelled-out numbers of a text as digits.
func (n *normalizer) numbers(text string) string {
	text = n.spaced(text)
	text = strings.Join(n.read(strings.Fields(text)), " ")
	return n.finished(text)
}

// spaced prepares a text for the number reader: "and a half" after a number
// word becomes "point five", a letter and a digit that touch are parted, and
// an ending that stands apart from its number joins it.
func (n *normalizer) spaced(text string) string {
	matches, err := n.half.FindAllStringIndex(text, -1)
	if err != nil {
		panic(fmt.Sprintf("englishnorm: %s: %v", n.half, err))
	}
	var parts []string
	start := 0
	for _, match := range matches {
		segment := text[start:match[0]]
		start = match[1]
		words := strings.Fields(segment)
		if len(words) == 0 {
			continue
		}
		parts = append(parts, segment)
		if last := words[len(words)-1]; n.decimals[last] || n.multipliers[last] != nil {
			parts = append(parts, halfAsDecimal)
		} else {
			parts = append(parts, halfAsWords)
		}
	}
	if rest := text[start:]; strings.TrimSpace(rest) != "" {
		parts = append(parts, rest)
	}
	text = strings.Join(parts, " ")
	for _, step := range n.numbered {
		text = step.apply(text)
	}
	return text
}

// finished writes an amount of money with its cents as one number, cents
// alone as cents, and a lone 1 as the word.
func (n *normalizer) finished(text string) string {
	text = replaced(n.cents, text, func(match regexp2.Match) string {
		cents := match.GroupByNumber(centsGroup).String()
		return match.GroupByNumber(currencyGroup).String() + match.GroupByNumber(unitsGroup).String() + "." + strings.Repeat("0", centDigits-len(cents)) + cents
	})
	text = replaced(n.loneCents, text, func(match regexp2.Match) string {
		// The pattern of cents alone captures only them.
		cents, _ := strconv.Atoi(match.GroupByNumber(1).String())
		return "¢" + strconv.Itoa(cents)
	})
	return n.one.apply(text)
}

func replaced(pattern *regexp2.Regexp, text string, with regexp2.MatchEvaluator) string {
	result, err := pattern.ReplaceFunc(text, with, 0, -1)
	if err != nil {
		panic(fmt.Sprintf("englishnorm: %s: %v", pattern, err))
	}
	return result
}

// read is the number reader: it goes through the words once, carrying the
// number read so far and the sign or currency to write before it, and gives
// the words back with each run of number words written as digits. Each case
// is that code's, in its order.
func (n *normalizer) read(words []string) []string {
	var (
		out    []string
		value  amount
		prefix string
		skip   bool
	)
	emit := func(text string) {
		out = append(out, prefix+text)
		prefix, value = "", amount{}
	}
	flush := func() {
		if value.held {
			emit(value.text())
		}
	}
	for index, current := range words {
		var previous, next string
		if index > 0 {
			previous = words[index-1]
		}
		if index+1 < len(words) {
			next = words[index+1]
		}
		if skip {
			skip = false
			continue
		}
		_, nextNumeric := fraction(next, true)
		sign, size := utf8.DecodeRuneInString(current)
		bare, signed := current, n.prefixes[sign]
		if signed {
			bare = current[size:]
		}
		ended := n.suffixed[current]
		switch number, numeric := fraction(bare, true); {
		case numeric:
			if value.held {
				if value.written() && strings.HasSuffix(value.digits, ".") {
					value = writtenAmount(value.digits + current)
					continue
				}
				flush()
			}
			if signed {
				prefix = string(sign)
			}
			if number.IsInt() {
				value = wholeAmount(new(big.Int).Set(number.Num()))
			} else {
				value = writtenAmount(bare)
			}
		case !n.words[current]:
			flush()
			emit(current)
		case n.zeros[current]:
			value = writtenAmount(value.orEmpty() + "0")
		case n.ones[current] != nil:
			ones := n.ones[current]
			switch small := ones.Cmp(big.NewInt(tenUnit)) < 0; {
			case !value.held:
				value = wholeAmount(ones)
			case value.written() || n.ones[previous] != nil:
				if n.tens[previous] != nil && small {
					value = writtenAmount(trimLast(value.text()) + ones.String())
				} else {
					value = writtenAmount(value.text() + ones.String())
				}
			case (small && value.endsOn(tenUnit)) || (!small && value.endsOn(hundredUnit)):
				value = wholeAmount(new(big.Int).Add(value.whole, ones))
			default:
				value = writtenAmount(value.text() + ones.String())
			}
		case n.onesEnded[current]:
			ones := ended.value
			switch small := ones.Cmp(big.NewInt(tenUnit)) < 0; {
			case !value.held:
				emit(ones.String() + ended.suffix)
			case value.written() || n.ones[previous] != nil:
				if n.tens[previous] != nil && small {
					emit(trimLast(value.text()) + ones.String() + ended.suffix)
				} else {
					emit(value.text() + ones.String() + ended.suffix)
				}
			case (small && value.endsOn(tenUnit)) || (!small && value.endsOn(hundredUnit)):
				emit(new(big.Int).Add(value.whole, ones).String() + ended.suffix)
			default:
				emit(value.text() + ones.String() + ended.suffix)
			}
		case n.tens[current] != nil:
			tens := n.tens[current]
			switch {
			case !value.held:
				value = wholeAmount(tens)
			case !value.written() && value.endsOn(hundredUnit):
				value = wholeAmount(new(big.Int).Add(value.whole, tens))
			default:
				value = writtenAmount(value.text() + tens.String())
			}
		case n.tensEnded[current]:
			tens := ended.value
			switch {
			case !value.held:
				emit(tens.String() + ended.suffix)
			case !value.written() && value.endsOn(hundredUnit):
				emit(new(big.Int).Add(value.whole, tens).String() + ended.suffix)
			default:
				emit(value.text() + tens.String() + ended.suffix)
			}
		case n.multipliers[current] != nil:
			multiplier := n.multipliers[current]
			switch {
			case !value.held:
				value = wholeAmount(multiplier)
			case value.written() || value.whole.Sign() == 0:
				if product, whole := scaled(value.text(), multiplier); whole {
					value = wholeAmount(product)
				} else {
					flush()
					value = wholeAmount(multiplier)
				}
			default:
				value = wholeAmount(grouped(value.whole, multiplier))
			}
		case ended.value != nil:
			multiplier := ended.value
			switch {
			case !value.held:
				emit(multiplier.String() + ended.suffix)
			case value.written():
				if product, whole := scaled(value.text(), multiplier); whole {
					emit(product.String() + ended.suffix)
				} else {
					flush()
					emit(multiplier.String() + ended.suffix)
				}
			default:
				emit(grouped(value.whole, multiplier).String() + ended.suffix)
			}
		case n.preceding[current] != "":
			flush()
			if n.words[next] || nextNumeric {
				prefix = n.preceding[current]
			} else {
				emit(current)
			}
		case n.following[current] != "":
			if value.held {
				prefix = n.following[current]
				emit(value.text())
			} else {
				emit(current)
			}
		case n.suffixers[current] != "":
			if value.held {
				emit(value.text() + n.suffixers[current])
			} else {
				emit(current)
			}
		case n.splitSuffixers[current] != nil:
			symbol, joins := n.splitSuffixers[current][next]
			switch {
			case !value.held:
				emit(current)
			case joins:
				emit(value.text() + symbol)
				skip = true
			default:
				flush()
				emit(current)
			}
		case !n.words[next] && !nextNumeric:
			flush()
			emit(current)
		case current == joiningWord:
			if n.multipliers[previous] == nil {
				flush()
				emit(current)
			}
		case n.repeats[current] != 0:
			one := n.ones[next]
			if one == nil && !n.zeros[next] {
				flush()
				emit(current)
				continue
			}
			repeated := "0"
			if one != nil {
				repeated = one.String()
			}
			value = writtenAmount(value.orEmpty() + strings.Repeat(repeated, n.repeats[current]))
			skip = true
		case current == pointWord:
			if n.decimals[next] || nextNumeric {
				value = writtenAmount(value.orEmpty() + ".")
			}
		}
	}
	flush()
	return out
}
