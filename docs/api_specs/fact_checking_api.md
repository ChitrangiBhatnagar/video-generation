# Fact-Checking API Specification

## Overview

The Fact-Checking API is a critical component of the PIB-VideoGen system that validates factual claims in text scripts against trusted sources. This document outlines the API endpoints, request/response formats, and integration guidelines.

## API Endpoints

### 1. Extract Claims

```
POST /api/v1/fact-check/extract-claims
```

#### Description
Extracts factual claims from input text that require verification.

#### Request

```json
{
  "text": "The Prime Minister inaugurated the new solar plant on June 15, 2023, which will generate 500 MW of clean energy for Delhi.",
  "language": "en",
  "claim_types": ["dates", "figures", "locations", "people", "organizations"],
  "confidence_threshold": 0.7
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| text | string | Yes | The input text to extract claims from |
| language | string | No | ISO language code (default: auto-detect) |
| claim_types | array | No | Types of claims to extract (default: all) |
| confidence_threshold | float | No | Minimum confidence score for extraction (0.0-1.0, default: 0.7) |

#### Response

```json
{
  "claims": [
    {
      "id": "claim_001",
      "text": "Prime Minister inaugurated the new solar plant",
      "type": "event",
      "entities": [
        {
          "text": "Prime Minister",
          "type": "person",
          "confidence": 0.95
        },
        {
          "text": "solar plant",
          "type": "facility",
          "confidence": 0.92
        }
      ],
      "confidence": 0.89
    },
    {
      "id": "claim_002",
      "text": "inauguration happened on June 15, 2023",
      "type": "date",
      "entities": [
        {
          "text": "June 15, 2023",
          "type": "date",
          "normalized": "2023-06-15",
          "confidence": 0.98
        }
      ],
      "confidence": 0.95
    },
    {
      "id": "claim_003",
      "text": "solar plant will generate 500 MW of clean energy",
      "type": "figure",
      "entities": [
        {
          "text": "500 MW",
          "type": "measurement",
          "value": 500,
          "unit": "MW",
          "confidence": 0.97
        }
      ],
      "confidence": 0.92
    },
    {
      "id": "claim_004",
      "text": "energy for Delhi",
      "type": "location",
      "entities": [
        {
          "text": "Delhi",
          "type": "location",
          "geo": {
            "lat": 28.6139,
            "lng": 77.2090
          },
          "confidence": 0.96
        }
      ],
      "confidence": 0.90
    }
  ],
  "metadata": {
    "total_claims": 4,
    "processing_time_ms": 156
  }
}
```

### 2. Verify Claims

```
POST /api/v1/fact-check/verify-claims
```

#### Description
Verifies extracted claims against trusted sources and databases.

#### Request

```json
{
  "claims": [
    {
      "id": "claim_001",
      "text": "Prime Minister inaugurated the new solar plant",
      "type": "event"
    },
    {
      "id": "claim_002",
      "text": "inauguration happened on June 15, 2023",
      "type": "date",
      "entities": [
        {
          "text": "June 15, 2023",
          "type": "date",
          "normalized": "2023-06-15"
        }
      ]
    },
    {
      "id": "claim_003",
      "text": "solar plant will generate 500 MW of clean energy",
      "type": "figure",
      "entities": [
        {
          "text": "500 MW",
          "type": "measurement",
          "value": 500,
          "unit": "MW"
        }
      ]
    },
    {
      "id": "claim_004",
      "text": "energy for Delhi",
      "type": "location",
      "entities": [
        {
          "text": "Delhi",
          "type": "location"
        }
      ]
    }
  ],
  "sources": ["pib_database", "ministry_websites", "official_records", "trusted_news"],
  "verification_level": "standard",
  "max_age_days": 30
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| claims | array | Yes | Array of claim objects to verify |
| sources | array | No | Trusted sources to check against (default: all available) |
| verification_level | string | No | Level of verification: "basic", "standard", or "thorough" (default: "standard") |
| max_age_days | integer | No | Maximum age of reference data in days (default: 30) |

#### Response

```json
{
  "verified_claims": [
    {
      "id": "claim_001",
      "text": "Prime Minister inaugurated the new solar plant",
      "verification_status": "verified",
      "confidence": 0.95,
      "sources": [
        {
          "name": "PIB Press Release",
          "id": "PIB20230615001",
          "url": "https://pib.gov.in/PressReleasePage.aspx?PRID=1234567",
          "timestamp": "2023-06-15T14:30:00Z",
          "excerpt": "The Prime Minister inaugurated the new solar plant in Delhi today..."
        },
        {
          "name": "Ministry of Power",
          "id": "MOP20230615003",
          "url": "https://powermin.gov.in/en/content/pm-inaugurates-delhi-solar-plant",
          "timestamp": "2023-06-15T16:45:00Z"
        }
      ]
    },
    {
      "id": "claim_002",
      "text": "inauguration happened on June 15, 2023",
      "verification_status": "verified",
      "confidence": 0.98,
      "sources": [
        {
          "name": "PIB Press Release",
          "id": "PIB20230615001",
          "url": "https://pib.gov.in/PressReleasePage.aspx?PRID=1234567",
          "timestamp": "2023-06-15T14:30:00Z"
        }
      ]
    },
    {
      "id": "claim_003",
      "text": "solar plant will generate 500 MW of clean energy",
      "verification_status": "partially_verified",
      "confidence": 0.75,
      "sources": [
        {
          "name": "Ministry of Power",
          "id": "MOP20230615003",
          "url": "https://powermin.gov.in/en/content/pm-inaugurates-delhi-solar-plant",
          "timestamp": "2023-06-15T16:45:00Z",
          "excerpt": "The solar plant has a capacity of 550 MW...",
          "discrepancy": "Capacity is 550 MW, not 500 MW"
        }
      ],
      "notes": "Claim states 500 MW but official source indicates 550 MW capacity"
    },
    {
      "id": "claim_004",
      "text": "energy for Delhi",
      "verification_status": "verified",
      "confidence": 0.92,
      "sources": [
        {
          "name": "PIB Press Release",
          "id": "PIB20230615001",
          "url": "https://pib.gov.in/PressReleasePage.aspx?PRID=1234567",
          "timestamp": "2023-06-15T14:30:00Z"
        },
        {
          "name": "Delhi Power Department",
          "id": "DPD20230616002",
          "url": "https://power.delhi.gov.in/news/new-solar-plant-connected-grid",
          "timestamp": "2023-06-16T09:15:00Z"
        }
      ]
    }
  ],
  "summary": {
    "total_claims": 4,
    "verified": 3,
    "partially_verified": 1,
    "unverified": 0,
    "conflicting": 0,
    "overall_confidence": 0.90,
    "requires_human_review": true,
    "review_reason": "Discrepancy in power generation capacity"
  },
  "metadata": {
    "processing_time_ms": 1250,
    "sources_checked": 4,
    "verification_level": "standard"
  }
}
```

### 3. Get Verification Status

```
GET /api/v1/fact-check/status/{verification_id}
```

#### Description
Retrieve the status of an ongoing or completed verification process.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| verification_id | string | Yes | The ID of the verification process |

#### Response

```json
{
  "verification_id": "ver_20230720_001",
  "status": "completed",
  "progress": 100,
  "started_at": "2023-07-20T10:15:30Z",
  "completed_at": "2023-07-20T10:15:45Z",
  "result_url": "/api/v1/fact-check/results/ver_20230720_001",
  "metadata": {
    "total_claims": 4,
    "verified": 3,
    "partially_verified": 1,
    "unverified": 0,
    "conflicting": 0
  }
}
```

### 4. Get Source Information

```
GET /api/v1/fact-check/sources
```

#### Description
Retrieve information about available trusted sources for fact-checking.

#### Response

```json
{
  "sources": [
    {
      "id": "pib_database",
      "name": "PIB Press Releases Database",
      "description": "Official press releases from Press Information Bureau",
      "coverage": {
        "start_date": "2018-01-01",
        "end_date": "current",
        "update_frequency": "daily"
      },
      "access_level": "full",
      "reliability_score": 0.99
    },
    {
      "id": "ministry_websites",
      "name": "Government Ministry Websites",
      "description": "Official websites of Indian government ministries",
      "coverage": {
        "start_date": "2020-01-01",
        "end_date": "current",
        "update_frequency": "daily"
      },
      "access_level": "full",
      "reliability_score": 0.95
    },
    {
      "id": "official_records",
      "name": "Government Official Records",
      "description": "Official government records and documents",
      "coverage": {
        "start_date": "2015-01-01",
        "end_date": "current",
        "update_frequency": "weekly"
      },
      "access_level": "restricted",
      "reliability_score": 0.98
    },
    {
      "id": "trusted_news",
      "name": "Verified News Sources",
      "description": "Curated list of verified news sources",
      "coverage": {
        "start_date": "2021-01-01",
        "end_date": "current",
        "update_frequency": "daily"
      },
      "access_level": "partial",
      "reliability_score": 0.85
    }
  ],
  "metadata": {
    "total_sources": 4,
    "last_updated": "2023-07-19T00:00:00Z"
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid claim format provided",
    "details": [
      {
        "field": "claims[2].entities[0].type",
        "issue": "Unknown entity type 'statistic', allowed types are: 'measurement', 'count', 'percentage', 'currency'"
      }
    ],
    "request_id": "req_7f6d59a2"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| invalid_request | 400 | The request was malformed or contained invalid parameters |
| unauthorized | 401 | Authentication credentials are missing or invalid |
| forbidden | 403 | The authenticated user does not have permission to access the resource |
| not_found | 404 | The requested resource was not found |
| rate_limit_exceeded | 429 | The rate limit for the API has been exceeded |
| internal_error | 500 | An unexpected error occurred on the server |
| service_unavailable | 503 | The service is temporarily unavailable |

## Authentication

All API requests must include an API key in the request header:

```
X-API-Key: your_api_key_here
```

For sensitive operations, additional JWT-based authentication is required:

```
Authorization: Bearer your_jwt_token_here
```

## Rate Limiting

The API enforces the following rate limits:

- 100 requests per minute per API key for extraction endpoints
- 50 requests per minute per API key for verification endpoints
- 1000 requests per day per API key for all endpoints combined

Rate limit information is included in the response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1627834520
```

## Integration Guidelines

### Recommended Integration Flow

1. **Extract Claims**: First, extract factual claims from the input text using the `/extract-claims` endpoint.
2. **Verify Claims**: Send the extracted claims to the `/verify-claims` endpoint for verification against trusted sources.
3. **Handle Results**: Process the verification results based on the verification status:
   - If all claims are `verified`, proceed with video generation.
   - If any claims are `unverified` or `conflicting`, flag them for human review.
   - If claims are `partially_verified`, decide based on the confidence score and discrepancy severity.

### Emergency Mode Integration

For urgent content (emergency alerts), use a streamlined verification process:

1. Set `verification_level` to `basic` to prioritize speed over thoroughness.
2. Set a lower `confidence_threshold` (e.g., 0.6) to reduce false negatives.
3. Implement a fallback mechanism to proceed with generation if verification takes longer than a predefined timeout (e.g., 5 seconds).

### Caching Strategy

Implement caching for frequently verified claims to improve performance:

1. Cache verification results with a TTL based on the claim type (e.g., 24 hours for historical facts, 1 hour for current events).
2. Include a cache key in the request to bypass verification for previously verified claims.
3. Implement cache invalidation when new information becomes available in the trusted sources.

## Appendix

### Supported Claim Types

| Type | Description | Examples |
|------|-------------|----------|
| date | Temporal information | "June 15, 2023", "last week", "2022-2023 fiscal year" |
| figure | Numerical data | "500 MW", "₹10,000 crore", "15% increase" |
| location | Geographical information | "Delhi", "Rajasthan", "Eastern India" |
| person | Individual names | "Prime Minister", "Chief Minister", "Secretary" |
| organization | Entity names | "Ministry of Power", "NITI Aayog", "Indian Railways" |
| event | Occurrences | "inauguration", "launch", "meeting", "conference" |
| policy | Government policies | "National Education Policy", "PM Kisan Scheme" |

### Verification Status Definitions

| Status | Description |
|--------|-------------|
| verified | Claim is fully supported by trusted sources |
| partially_verified | Claim is mostly supported but has minor discrepancies |
| unverified | Claim cannot be verified with available sources |
| conflicting | Claim contradicts information from trusted sources |
| outdated | Claim was true but is no longer accurate |

### Confidence Score Guidelines

| Score Range | Interpretation | Recommended Action |
|-------------|----------------|--------------------|
| 0.9 - 1.0 | Very high confidence | Proceed without review |
| 0.8 - 0.9 | High confidence | Proceed with minimal review |
| 0.7 - 0.8 | Moderate confidence | Review discrepancies |
| 0.6 - 0.7 | Low confidence | Thorough review required |
| < 0.6 | Very low confidence | Do not proceed, requires human verification |