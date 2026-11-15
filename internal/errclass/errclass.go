// Package errclass implements FR-009 error classification and the
// resolved §7 retry semantics (07-open-questions.md).
package errclass

import (
	"regexp"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginabi"
)

// Class is a stable error-category identifier (FR-009) used as the
// envelope error Code.
type Class string

const (
	ClassAuth         Class = "auth_failure"
	ClassInvalidModel Class = "invalid_model"
	ClassUnsupported  Class = "unsupported_protocol_or_parameter"
	ClassRateLimit    Class = "rate_limit"
	ClassQuota        Class = "quota_exhaustion"
	ClassBilling      Class = "billing"
	ClassUpstream     Class = "upstream_server_failure"
	ClassNetwork      Class = "timeout_or_network_failure"
	ClassTranslation  Class = "translation_failure"
)

// Error describes classified request errors and retryability for CPA execution.
type Error struct {
	Class      Class
	Message    string
	StatusCode int
	Retryable  bool
	// RetryAfterSeconds is the upstream's delta-seconds retry hint. v8's
	// conductor times the credential cooldown from it and mirrors it into
	// the client's Retry-After header; zero means "no hint".
	RetryAfterSeconds int
	// CredentialScoped marks a failure that applies to the whole credential
	// across all models (an account-wide 429 or spend limit) rather than only
	// the requested model. v8 reads it together with a 429 status.
	CredentialScoped bool
}

func (e *Error) Error() string {
	return string(e.Class) + ": " + e.Message
}

// IsCloudflareChallengeText reports whether an upstream error text carries
// Cloudflare challenge vocabulary. The predicate is deliberately close to the
// host's own matcher (conductor_cooldown.go): the bare phrase, the canonical
// "just a moment" title, or "attention required" — the other standard
// challenge page title — but ONLY when "cloudflare" co-occurs with it, since
// a plain credential-403 JSON could carry that phrase in its message and a
// false positive here delays credential rotation to a short backoff.
// "just a moment" stays standalone because the challenge title sits within
// the first bytes of the page while the Cloudflare branding may fall past the
// bounded scan window.
func IsCloudflareChallengeText(message string) bool {
	lower := strings.ToLower(message)
	if strings.Contains(lower, "cloudflare challenge") ||
		strings.Contains(lower, "just a moment") ||
		strings.Contains(lower, "challenge-platform") ||
		strings.Contains(lower, "cf-mitigated") {
		return true
	}
	return strings.Contains(lower, "attention required") && strings.Contains(lower, "cloudflare")
}

// fromCloudflareChallenge reclassifies a 403 whose *whole* error text carries
// challenge vocabulary. The host routes challenge results into a short
// backoff instead of burning the credential, but it only recognizes its own
// vocabulary — and a real challenge page puts its markers ("Just a
// moment...", "challenge-platform") deep in the HTML, past the 80-character
// snippet callers keep. So classification inspects the full text it was
// handed and prefixes the host phrase onto the bounded snippet (never
// widening it, FR-011).
func fromCloudflareChallenge(status int, msg string) *Error {
	e := &Error{Class: ClassUpstream, Message: msg, StatusCode: status, Retryable: true}
	if !strings.Contains(strings.ToLower(msg), "cloudflare challenge") {
		e.Message = "cloudflare challenge: " + msg
	}
	return e
}

// FromStatus classifies an upstream HTTP status per resolved §7.
// Network/timeout failures (no status) use FromNetwork instead. A 403 whose
// full message carries Cloudflare challenge vocabulary is a WAF interstitial,
// not a credential problem: it becomes ClassUpstream + retryable and is
// prefixed with the host's "cloudflare challenge" phrase. A redacted message
// carrying quota-exhaustion vocabulary ("quota", or the "usage limit" wording
// ollama.com actually returns for a monthly usage cap: "you (user) have
// reached your monthly usage limit, add ...") upgrades 429/402/403 to
// ClassQuota. For 429 that additionally sets CredentialScoped so the host
// treats it as whole-credential quota (reason "credential_quota") instead of
// the generic 429 ladder; 402/403 stay unflagged — the host's 402 branch is
// already a fixed credential-wide 30-minute cooldown, and the credential-
// quota routing is documented for 429 only.
func FromStatus(status int, message string) *Error {
	msg := Redact(message)
	if status == 403 && IsCloudflareChallengeText(msg) {
		return fromCloudflareChallenge(status, msg)
	}
	lowerMsg := strings.ToLower(msg)
	// Window-exhaustion vocabulary. "quota" is the legacy generic term;
	// "usage limit" covers the observed monthly/weekly caps (two live
	// samples); "session window"/"session limit" is preemptive for the
	// Ollama Pro 5-hour session window (whose account-side datum is the
	// balance API's Session window the predictor already consumes) — the
	// real 429 wording is still unobserved here, so tighten to the sample
	// once one lands. Plan structure (per operator-confirmed upstream
	// behavior): Max carries only a monthly quota; Pro carries a weekly
	// quota and the 5h session window as two INDEPENDENT budgets, each
	// prompting its own exhaustion message — a session-window 429 arrives
	// while weekly balance remains, and vice versa. Both are window
	// budgets recoverable only on reset (or payment/upgrade), never by
	// immediate retry, so treating all of them as account-level quota
	// (whole-credential cooldown, 30-min probes) cannot hurt a genuinely
	// transient rate limit unless the upstream invents a window name with
	// different semantics.
	quota := strings.Contains(lowerMsg, "quota") ||
		strings.Contains(lowerMsg, "usage limit") ||
		strings.Contains(lowerMsg, "session window") ||
		strings.Contains(lowerMsg, "session limit")
	quotaClass := func(base Class) Class {
		if quota {
			return ClassQuota
		}
		return base
	}
	switch status {
	case 401:
		return &Error{Class: ClassAuth, Message: msg, StatusCode: status, Retryable: true}
	case 403:
		return &Error{Class: quotaClass(ClassAuth), Message: msg, StatusCode: status, Retryable: true}
	case 402:
		return &Error{Class: quotaClass(ClassBilling), Message: msg, StatusCode: status, Retryable: true}
	case 404:
		return &Error{Class: ClassInvalidModel, Message: msg, StatusCode: status}
	case 429:
		return &Error{Class: quotaClass(ClassRateLimit), Message: msg, StatusCode: status, Retryable: true, CredentialScoped: quota}
	}
	if status >= 500 {
		return &Error{Class: ClassUpstream, Message: msg, StatusCode: status, Retryable: true}
	}
	return &Error{Class: ClassUnsupported, Message: msg, StatusCode: status}
}

// FromNetwork classifies a PRE-first-byte network/timeout failure; always
// retryable (§7). Failures observed after the first upstream byte must use
// PostFirstByteNetwork instead.
func FromNetwork(err error) *Error {
	msg := ""
	if err != nil {
		msg = Redact(err.Error())
	}
	// 502: the host maps HTTPStatus 0 to a generic 500 before its
	// default recoverable-failure branch; an explicit gateway status keeps
	// the transport failure distinguishable in client responses while
	// hitting the same credential-failure treatment.
	return &Error{Class: ClassNetwork, Message: msg, StatusCode: 502, Retryable: true}
}

// PostFirstByteNetwork classifies a network/timeout failure seen AFTER the
// first upstream byte was received: part of the response may already be
// delivered downstream, so a retry would risk duplicate delivery of a half
// stream — never retryable (§7).
func PostFirstByteNetwork(msg string) *Error {
	return &Error{Class: ClassNetwork, Message: Redact(msg), Retryable: false}
}

// Translation reports an FR-005/FR-006 translation failure; never retryable.
func Translation(msg string) *Error {
	// 400: a translation failure is a property of the client request body.
	// Keeping the status explicit stops the host from reading the missing
	// status as a 500 and penalizing the selected credential (v8's
	// clienterror.IsRequestFault exempts 4xx from cooldown/rotation).
	return &Error{Class: ClassTranslation, Message: Redact(msg), StatusCode: 400}
}

// UpstreamFallback builds a ClassUpstream error for the adapter fallback
// sites (chunk/failure errors) that have no upstream status code. Resolved
// §7 makes upstream server failures retryable; the message is redacted at
// construction.
func UpstreamFallback(msg string) *Error {
	// 502: an explicit gateway-failure status keeps the host from reading
	// the missing status as a generic 500; both map to the host's default
	// recoverable-failure branch, but 502 is honest in client responses.
	return &Error{Class: ClassUpstream, Message: Redact(msg), StatusCode: 502, Retryable: true}
}

// ToEnvelopeError converts to the shared SDK wire type with a redacted
// message. Re-redacting at the envelope edge is INTENTIONAL defense-in-depth:
// constructors already redact (so non-envelope consumers — logs, executor
// fail paths reading .Message directly) are safe, and Redact is idempotent.
//
// Retryable rides along only because pre-v8 hosts read it; v8's routing and
// cooldown pipeline is driven by Code (the class, or a recognized
// pluginabi.ErrorCode* value), HTTPStatus, RetryAfterSeconds, and
// CredentialScoped instead.
func ToEnvelopeError(e *Error) pluginabi.Error {
	return pluginabi.Error{
		Code:              string(e.Class),
		Message:           Redact(e.Message),
		Retryable:         e.Retryable,
		HTTPStatus:        e.StatusCode,
		RetryAfterSeconds: e.RetryAfterSeconds,
		CredentialScoped:  e.CredentialScoped,
	}
}

// bearerRe matches a bearer token (case-insensitive scheme, optional
// space/tab separator, token terminated by whitespace, comma, or quote).
// The separator is optional so a bare "bearer" word is redacted exactly as
// the previous hand-rolled scanner did. Redact is idempotent — applying it
// twice yields the same output — which is what makes edge re-redaction safe.
var bearerRe = regexp.MustCompile(`(?i)bearer[ \t]*[^\s,"]*`)

// apiKeyRe finds x-api-key header echoes (Messages-route auth, FR-007):
// case-insensitive name, then a run of quote/colon/equals/space separators,
// then the value run. The value class spans hex/base62 plus the base64
// alphabet ("+", "/", "="), matching bearerRe/queryKeyRe sibling classes.
// The candidate is only replaced when the value looks
// like a key (contains a digit) so ordinary prose after the header name —
// "x-api-key authentication failed" — passes untouched; every real key
// alphabet (hex, base64, base62) carries digits. Idempotent: "[redacted]"
// starts with '[' which is outside the value class, so it never re-matches.
var apiKeyRe = regexp.MustCompile(`(?i)x-api-key["':= ]*[A-Za-z0-9_\-.+/=]+`)

func redactAPIKey(match string) string {
	value := strings.TrimLeft(match[len("x-api-key"):], "\"':= ")
	hasDigit := false
	for _, r := range value {
		if r >= '0' && r <= '9' {
			hasDigit = true
			break
		}
	}
	if !hasDigit || len(value) < 8 {
		return match
	}
	return match[:len(match)-len(value)] + "[redacted]"
}

// queryKeyRe redacts secrets carried in URL query strings (FR-011): transport
// failures conventionally echo the full URL, so base-url/catalog-url keys
// passed as ?api_key=… / ?token=… must not survive into envelopes or logs.
// The value run stops at whitespace, '&', and quotes so consecutive params
// survive; the replacement keeps the parameter name. Idempotent: the
// redacted value "[redacted]" still matches the value class, so re-running
// yields the same text.
var queryKeyRe = regexp.MustCompile(`(?i)([?&](?:api[_-]?key|key|access[_-]?token|token)=)[^\s&"']+`)

// Redact strips bearer tokens, x-api-key echoes, and query-string key
// parameters from a message so keys never reach logs or clients.
func Redact(msg string) string {
	msg = bearerRe.ReplaceAllString(msg, "Bearer [redacted]")
	msg = apiKeyRe.ReplaceAllStringFunc(msg, redactAPIKey)
	return queryKeyRe.ReplaceAllString(msg, "${1}[redacted]")
}
