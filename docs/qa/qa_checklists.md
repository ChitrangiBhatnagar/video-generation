# QA Checklists and Human Review Guidelines

## Overview

This document provides comprehensive quality assurance checklists and human review guidelines for the PIB-VideoGen system. These guidelines ensure that all generated videos meet the Press Information Bureau's standards for quality, accuracy, accessibility, and compliance before public release.

## Video Quality Checklist

### Visual Quality Assessment

| # | Check Item | Pass Criteria | Priority |
|---|-----------|---------------|----------|
| 1 | Resolution | 1280×720 px or higher with clear details | Critical |
| 2 | Frame rate | Consistent 24-30 fps without drops | High |
| 3 | Color accuracy | Official PIB color palette, proper white balance | High |
| 4 | Contrast & brightness | Readable text, visible details in highlights/shadows | Medium |
| 5 | Sharpness | No blurring of important elements | High |
| 6 | Artifacts | No visible compression artifacts, banding, or pixelation | High |
| 7 | Flicker | No temporal flickering between frames | Critical |
| 8 | Temporal coherence | Smooth transitions between frames (score > 0.9) | Critical |
| 9 | Image alignment | Proper framing of key subjects | Medium |
| 10 | Logo placement | PIB seal clearly visible, correct size and position | Critical |

### Audio Quality Assessment (If Applicable)

| # | Check Item | Pass Criteria | Priority |
|---|-----------|---------------|----------|
| 1 | Audio clarity | Clear speech without distortion | Critical |
| 2 | Volume levels | Consistent volume throughout (-14 LUFS target) | High |
| 3 | Background noise | Minimal background noise (SNR > 20dB) | Medium |
| 4 | Sync with visuals | Audio properly synchronized with video | Critical |
| 5 | Audio format | AAC encoding at minimum 128 kbps | Medium |

## Content Accuracy Checklist

### Factual Verification

| # | Check Item | Pass Criteria | Priority |
|---|-----------|---------------|----------|
| 1 | Dates and times | All dates/times match official records | Critical |
| 2 | Names and titles | Correct spelling and designation of all individuals | Critical |
| 3 | Locations | Accurate geographical references | High |
| 4 | Statistics & figures | Numbers match official sources | Critical |
| 5 | Policy details | Accurate representation of government policies | Critical |
| 6 | Event descriptions | Correct depiction of events and timelines | High |
| 7 | Source verification | All facts verified against trusted sources | Critical |
| 8 | "⚠ Check Required" flags | All flagged items manually verified | Critical |

### Text and Language

| # | Check Item | Pass Criteria | Priority |
|---|-----------|---------------|----------|
| 1 | Spelling | No spelling errors | High |
| 2 | Grammar | Correct grammar throughout | High |
| 3 | Terminology | Consistent use of official terminology | Medium |
| 4 | Language detection | Correct language identification for multi-language content | High |
| 5 | Translation accuracy | Accurate translations for multi-language content | Critical |
| 6 | Text length | Text fits within frame and is readable | High |
| 7 | Abbreviations | Proper use and explanation of abbreviations | Medium |
| 8 | Clarity | Clear and concise messaging | High |

## Accessibility Checklist

| # | Check Item | Pass Criteria | Priority |
|---|-----------|---------------|----------|
| 1 | Subtitle timing | ≥ 4s display time per subtitle | Critical |
| 2 | Subtitle contrast | Meets WCAG 2.1 AA contrast requirements (4.5:1) | Critical |
| 3 | Font size | Minimum 24px at 1280×720 resolution | High |
| 4 | Reading speed | Maximum 20 characters per second | High |
| 5 | Color blindness | Content distinguishable for color blind viewers | High |
| 6 | Alt text | Available for all key visual elements | Medium |
| 7 | Audio description | Available for visually impaired users (if applicable) | Medium |
| 8 | SRT file | Accurate, synchronized subtitle file included | Critical |
| 9 | Screen reader compatibility | Text elements accessible to screen readers | High |
| 10 | Keyboard navigation | Video player supports keyboard controls | Medium |

## Brand Compliance Checklist

| # | Check Item | Pass Criteria | Priority |
|---|-----------|---------------|----------|
| 1 | PIB seal | Correct version of seal used | Critical |
| 2 | Color scheme | Adherence to PIB color palette | High |
| 3 | Typography | Official fonts used consistently | High |
| 4 | Logos | Correct usage of ministry/department logos | Critical |
| 5 | Watermarks | "Official Use Only" watermark when required | Critical |
| 6 | Opening/closing sequences | Standard PIB intro/outro when applicable | Medium |
| 7 | Attributions | Proper credits and attributions | Medium |
| 8 | Tone | Appropriate tone for government communication | High |

## Technical Compliance Checklist

| # | Check Item | Pass Criteria | Priority |
|---|-----------|---------------|----------|
| 1 | File format | MP4 with H.264 encoding | High |
| 2 | File size | Optimized for distribution channels | Medium |
| 3 | Metadata | Complete and accurate video metadata | Medium |
| 4 | Duration | 16-30 seconds (unless otherwise specified) | High |
| 5 | FVD score | < 200 | High |
| 6 | IS score | > 70 | High |
| 7 | Temporal coherence | > 0.9 | Critical |
| 8 | Playback compatibility | Works on standard government platforms | Critical |
| 9 | Security | No embedded malicious content | Critical |

## Human Review Process

### Review Workflow

1. **Automated QA**
   - System performs initial automated checks
   - Flags potential issues for human review
   - Calculates quality metrics (FVD, IS, temporal coherence)

2. **Technical Review**
   - Technical reviewer assesses video quality and technical compliance
   - Verifies all automated metrics
   - Checks for artifacts and visual issues
   - Approves or returns for regeneration (up to 3 times)

3. **Content Review**
   - Content specialist verifies factual accuracy
   - Reviews all "⚠ Check Required" flags
   - Ensures message clarity and effectiveness
   - Approves or requests revisions

4. **Accessibility Review**
   - Accessibility expert verifies WCAG 2.1 AA compliance
   - Reviews subtitles and text elements
   - Ensures all accessibility features are properly implemented
   - Approves or requests accessibility improvements

5. **Final Approval**
   - Senior reviewer performs final check
   - Verifies all previous approvals
   - Authorizes public release

### Review Roles and Responsibilities

| Role | Responsibilities | Required Expertise |
|------|-----------------|--------------------|
| Technical Reviewer | Video quality, technical standards | Video production, technical QA |
| Content Specialist | Factual accuracy, messaging | Subject matter expertise, government communications |
| Accessibility Expert | Accessibility compliance | WCAG standards, accessibility testing |
| Senior Reviewer | Final approval, policy compliance | PIB policies, government communications |

### Review Timeframes

| Priority | Maximum Review Time | Use Cases |
|----------|---------------------|----------|
| Emergency | 15 minutes | Disaster alerts, critical public safety information |
| Urgent | 2 hours | Time-sensitive announcements, breaking news |
| Standard | 24 hours | Regular policy updates, ceremonial announcements |
| Planned | 48 hours | Educational content, complex policy explanations |

## Special Case Guidelines

### Emergency Content Review

For content flagged as `urgent: true`:

1. Prioritize speed while maintaining minimum quality standards
2. Focus on factual accuracy of critical information
3. Ensure accessibility of key emergency instructions
4. Allow simplified branding for faster production
5. Document emergency approval in the review log

### Sensitive Content Guidelines

For content containing sensitive information:

1. Verify appropriate security classification
2. Ensure proper handling of personally identifiable information
3. Check for unintentional disclosure of sensitive locations
4. Verify appropriate blurring of faces when required
5. Confirm "Official Use Only" watermarking when applicable
6. Require additional approval from security team

### Multi-Language Content Review

For content with multiple languages:

1. Verify language detection accuracy for each segment
2. Ensure accurate translations by native speakers
3. Check subtitle synchronization for each language
4. Verify cultural appropriateness of visuals for target languages
5. Ensure consistent terminology across languages

## Review Documentation

### Review Form Template

```
PIB-VideoGen Review Form

Video ID: [VIDEO_ID]
Title: [TITLE]
Duration: [DURATION]
Use Case: [USE_CASE]
Priority: [PRIORITY]

Technical Review
- Reviewer: [NAME]
- Date/Time: [DATE_TIME]
- Quality Metrics: FVD=[SCORE], IS=[SCORE], TC=[SCORE]
- Issues Identified: [ISSUES]
- Recommendation: [APPROVE/REGENERATE/REJECT]
- Comments: [COMMENTS]

Content Review
- Reviewer: [NAME]
- Date/Time: [DATE_TIME]
- Fact Check Flags Resolved: [YES/NO]
- Issues Identified: [ISSUES]
- Recommendation: [APPROVE/REVISE/REJECT]
- Comments: [COMMENTS]

Accessibility Review
- Reviewer: [NAME]
- Date/Time: [DATE_TIME]
- WCAG 2.1 AA Compliant: [YES/NO]
- Issues Identified: [ISSUES]
- Recommendation: [APPROVE/IMPROVE/REJECT]
- Comments: [COMMENTS]

Final Approval
- Reviewer: [NAME]
- Date/Time: [DATE_TIME]
- Decision: [APPROVED/REJECTED]
- Comments: [COMMENTS]
```

### Review Log

Maintain a comprehensive review log with the following information:

- Video ID and metadata
- Review timestamps and durations
- Reviewer identifications
- Issues identified and resolutions
- Regeneration attempts and outcomes
- Final approval status
- Distribution channels and dates

## Quality Metrics and Thresholds

### Automated Metrics

| Metric | Acceptable Range | Critical Threshold | Measurement Method |
|--------|-----------------|-------------------|-------------------|
| FVD (Fréchet Video Distance) | < 200 | > 300 | Compare against reference videos |
| IS (Inception Score) | > 70 | < 50 | Evaluate visual quality and diversity |
| Temporal Coherence | > 0.9 | < 0.8 | Frame-to-frame consistency analysis |
| CLIP Score | > 0.8 | < 0.6 | Text-image alignment evaluation |
| Subtitle Sync | < 200ms offset | > 500ms offset | Automated subtitle timing analysis |

### Human Evaluation Metrics

| Aspect | Rating Scale | Minimum Acceptable | Evaluation Method |
|--------|-------------|-------------------|------------------|
| Visual Quality | 1-5 | 4.0 average | Panel review with standardized rubric |
| Factual Accuracy | 1-5 | 5.0 (no errors) | Fact-checking against official sources |
| Message Clarity | 1-5 | 4.0 average | Comprehension testing |
| Accessibility | 1-5 | 4.5 average | Accessibility expert evaluation |
| Brand Compliance | 1-5 | 4.5 average | Brand guidelines checklist |

## Continuous Improvement

### Feedback Loop

1. **Collect Review Data**
   - Aggregate review outcomes and metrics
   - Identify common issues and patterns
   - Track regeneration rates and causes

2. **Analysis**
   - Quarterly analysis of review data
   - Identify model strengths and weaknesses
   - Correlate issues with specific content types or parameters

3. **Model Refinement**
   - Provide feedback to training team
   - Prioritize improvements based on issue frequency and severity
   - Update λ parameters based on quality outcomes

4. **Process Improvement**
   - Refine review checklists based on emerging issues
   - Update guidelines to address new use cases
   - Optimize review workflow for efficiency

### Quarterly Audit

Conduct quarterly audits of the review process:

1. Random sampling of approved videos for re-review
2. Verification of review documentation completeness
3. Assessment of reviewer consistency and accuracy
4. Evaluation of review timeframes against targets
5. Identification of process improvement opportunities

## Appendix

### Visual Reference Guide

- Examples of acceptable vs. unacceptable video quality
- Proper PIB seal placement and sizing
- Correct subtitle formatting and positioning
- Approved color palettes and typography
- Examples of common artifacts to watch for

### Reviewer Training Requirements

- Initial certification requirements for each reviewer role
- Continuing education and recertification schedule
- Cross-training recommendations for backup coverage
- Specialized training for emergency content review

### Escalation Procedures

- Clear guidelines for when to escalate issues
- Contact information for escalation points
- Service level agreements for escalation response
- Documentation requirements for escalated issues

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|--------|
| 1.0 | [DATE] | [AUTHOR] | Initial document |
| 1.1 | [DATE] | [AUTHOR] | Added multi-language review guidelines |
| 1.2 | [DATE] | [AUTHOR] | Updated accessibility requirements to WCAG 2.1 AA |