use crate::settings::PostProcessProvider;
use log::debug;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, REFERER, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Clone)]
struct JsonSchema {
    name: String,
    strict: bool,
    schema: Value,
}

#[derive(Debug, Serialize, Clone)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    json_schema: JsonSchema,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningConfig>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

/// Detect whether a 400 error body is caused by the reasoning-control parameters.
///
/// Handy disables reasoning by default (`reasoning_effort: "none"` for custom
/// endpoints, `reasoning: { effort: "none", exclude: true }` for OpenRouter) to
/// cut post-processing latency. Some OpenAI-compatible endpoints reject that:
///   - OpenRouter GPT-OSS: "Reasoning is mandatory for this endpoint and cannot be disabled."
///   - DeepSeek: "reasoning_effort: unknown variant `none`, expected one of ..."
/// Both mention "reasoning", so we can retry with a valid setting instead of failing.
fn is_reasoning_error(body: &str) -> bool {
    body.to_lowercase().contains("reasoning")
}

/// Given the current reasoning parameters, produce the next (less aggressive)
/// variant to retry with, or `None` once there is nothing left to degrade.
///
/// Ladder: disabled ("none") -> minimal ("low", fastest that keeps reasoning on)
/// -> omitted entirely (let the endpoint use its default).
fn degrade_reasoning(
    reasoning_effort: Option<String>,
    reasoning: Option<ReasoningConfig>,
) -> Option<(Option<String>, Option<ReasoningConfig>)> {
    // Custom-style top-level field currently set to "none" -> retry with "low".
    if reasoning_effort.as_deref() == Some("none") {
        return Some((Some("low".to_string()), reasoning));
    }

    // OpenRouter-style nested object currently set to "none" -> retry with "low",
    // preserving `exclude` so reasoning tokens stay out of the structured response.
    if reasoning.as_ref().and_then(|r| r.effort.as_deref()) == Some("none") {
        let exclude = reasoning.and_then(|r| r.exclude);
        return Some((
            None,
            Some(ReasoningConfig {
                effort: Some("low".to_string()),
                exclude,
            }),
        ));
    }

    // Any other non-default value already tried -> drop reasoning params entirely.
    if reasoning_effort.is_some() || reasoning.is_some() {
        return Some((None, None));
    }

    None
}

/// Build headers for API requests based on provider type
fn build_headers(provider: &PostProcessProvider, api_key: &str) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();

    // Common headers
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        REFERER,
        HeaderValue::from_static("https://github.com/cjpais/Handy"),
    );
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Handy/1.0 (+https://github.com/cjpais/Handy)"),
    );
    headers.insert("X-Title", HeaderValue::from_static("Handy"));

    // Provider-specific auth headers
    if !api_key.is_empty() {
        if provider.id == "anthropic" {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(api_key)
                    .map_err(|e| format!("Invalid API key header value: {}", e))?,
            );
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        } else {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .map_err(|e| format!("Invalid authorization header value: {}", e))?,
            );
        }
    }

    Ok(headers)
}

/// Create an HTTP client with provider-specific headers
fn create_client(provider: &PostProcessProvider, api_key: &str) -> Result<reqwest::Client, String> {
    let headers = build_headers(provider, api_key)?;
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))
}

/// Send a chat completion request to an OpenAI-compatible API
/// Returns Ok(Some(content)) on success, Ok(None) if response has no content,
/// or Err on actual errors (HTTP, parsing, etc.)
pub async fn send_chat_completion(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    prompt: String,
    reasoning_effort: Option<String>,
    reasoning: Option<ReasoningConfig>,
) -> Result<Option<String>, String> {
    send_chat_completion_with_schema(
        provider,
        api_key,
        model,
        prompt,
        None,
        None,
        reasoning_effort,
        reasoning,
    )
    .await
}

/// Send a chat completion request with structured output support
/// When json_schema is provided, uses structured outputs mode
/// system_prompt is used as the system message when provided
/// reasoning_effort sets the OpenAI-style top-level field (e.g., "none", "low", "medium", "high")
/// reasoning sets the OpenRouter-style nested object (effort + exclude)
#[allow(clippy::too_many_arguments)]
pub async fn send_chat_completion_with_schema(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    user_content: String,
    system_prompt: Option<String>,
    json_schema: Option<Value>,
    reasoning_effort: Option<String>,
    reasoning: Option<ReasoningConfig>,
) -> Result<Option<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');
    let url = format!("{}/chat/completions", base_url);

    debug!("Sending chat completion request to: {}", url);

    let client = create_client(provider, &api_key)?;

    // Build messages vector
    let mut messages = Vec::new();

    // Add system prompt if provided
    if let Some(system) = system_prompt {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system,
        });
    }

    // Add user message
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: user_content,
    });

    // Build response_format if schema is provided
    let response_format = json_schema.map(|schema| ResponseFormat {
        format_type: "json_schema".to_string(),
        json_schema: JsonSchema {
            name: "transcription_output".to_string(),
            strict: true,
            schema,
        },
    });

    // Retry loop: if the endpoint rejects the reasoning-control parameters with a
    // 400, degrade them (none -> low -> omitted) and retry instead of failing.
    let mut cur_reasoning_effort = reasoning_effort;
    let mut cur_reasoning = reasoning;

    loop {
        let request_body = ChatCompletionRequest {
            model: model.to_string(),
            messages: messages.clone(),
            response_format: response_format.clone(),
            reasoning_effort: cur_reasoning_effort.clone(),
            reasoning: cur_reasoning.clone(),
        };

        let response = client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        let status = response.status();
        if status.is_success() {
            let completion: ChatCompletionResponse = response
                .json()
                .await
                .map_err(|e| format!("Failed to parse API response: {}", e))?;

            return Ok(completion
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone()));
        }

        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());

        if status.as_u16() == 400 && is_reasoning_error(&error_text) {
            if let Some((next_effort, next_reasoning)) =
                degrade_reasoning(cur_reasoning_effort.clone(), cur_reasoning.clone())
            {
                debug!(
                    "Endpoint rejected reasoning params (400); retrying with degraded reasoning: reasoning_effort={:?}, reasoning={:?}",
                    next_effort, next_reasoning
                );
                cur_reasoning_effort = next_effort;
                cur_reasoning = next_reasoning;
                continue;
            }
        }

        return Err(format!(
            "API request failed with status {}: {}",
            status, error_text
        ));
    }
}

/// Fetch available models from an OpenAI-compatible API
/// Returns a list of model IDs
pub async fn fetch_models(
    provider: &PostProcessProvider,
    api_key: String,
) -> Result<Vec<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');
    let url = format!("{}/models", base_url);

    debug!("Fetching models from: {}", url);

    let client = create_client(provider, &api_key)?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch models: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!(
            "Model list request failed ({}): {}",
            status, error_text
        ));
    }

    let parsed: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let mut models = Vec::new();

    // Handle OpenAI format: { data: [ { id: "..." }, ... ] }
    if let Some(data) = parsed.get("data").and_then(|d| d.as_array()) {
        for entry in data {
            if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                models.push(id.to_string());
            } else if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                models.push(name.to_string());
            }
        }
    }
    // Handle array format: [ "model1", "model2", ... ]
    else if let Some(array) = parsed.as_array() {
        for entry in array {
            if let Some(model) = entry.as_str() {
                models.push(model.to_string());
            }
        }
    }

    Ok(models)
}
